
/*
    Copyright(c) 2023 Addis Dittebrandt and Lorenzo Tessari
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

// accelerators/bvh.cpp*
#include "accelerators/bvhstochsubset.h"

#include <fstream>
#include <algorithm>
#include <fstream>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <tuple>

#include "interaction.h"
#include "parallel.h"
#include "paramset.h"
#include "shapes/triangle.h"
#include "stats.h"
#include "core/counters.h"

// random
#include "lowdiscrepancy.h"

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/BVH tree", treeBytes);
STAT_RATIO("BVH/Primitives per leaf node", totalPrimitives, totalLeafNodes);
STAT_COUNTER("BVH/Interior nodes", interiorNodes);
STAT_COUNTER("BVH/Leaf nodes", leafNodes);
STAT_COUNTER("BVH Node intersections", nodeIntersect);
STAT_COUNTER("Primitive intersections", primIntersect);

namespace bvhstochsubset {

void RecursiveBuildParameters::printParameters(int indent) {
    std::string spaces;
    for (int i = 0; i < indent; i++) spaces += " ";

    std::cout << "RecursiveBuildParameters:" << std::endl
        << spaces << "binCount: " << binCount << std::endl 
        << spaces << "maxPrimsInNode: " << maxPrimsInNode << std::endl
        << spaces << "ignoreLeafCost: " << ignoreLeafCost << std::endl;
}

void BuildParameters::printParameters(int indent) {
    std::string spaces;
    for (int i = 0; i < indent; i++) spaces += " ";

    std::cout << "BuildParameters:" << std::endl
        << spaces << "minSetSize: " << minSetSize << std::endl
        << spaces << "subsetSizeFrac: " << subsetSizeFrac << std::endl
        << spaces << "cdfKind: " << cdfKindName(cdfKind) << std::endl
        << spaces << "cdfPrecision: " << cdfPrecisionName(cdfPrecision) << std::endl
        << spaces << "weightClampingAlgorithm: " << weightClampingAlgorithmName(weightClampingAlgorithm) << std::endl
        << spaces << "uniformCdfFrac: " << uniformCdfFrac << std::endl
        << spaces << "samplingMethod: " << samplingMethodName(samplingMethod) << std::endl
        << spaces << "findBestNodeAlgorithm: " << findBestNodeAlgorithmName(findBestNodeAlgorithm) << std::endl
        << spaces << "localMortonSearchWindow: " << localMortonSearchWindow << std::endl;
    std::cout << spaces << "subsetBuildParameters: ";
    subsetBuildParameters.printParameters(indent + 4);
    
    std::cout << spaces << "baseBuildParameters: ";
    baseBuildParameters.printParameters(indent + 4);
        
    std::cout
        << spaces << "validateSubset: " << validateSubset << std::endl
        << spaces << "validateBestNodeForSubsetPrimitives: " << validateBestNodeForSubsetPrimitives << std::endl
        << spaces << "validateNonEmptyNodes: " << validateNonEmptyNodes << std::endl;
}

std::string cdfKindName(CdfKind cdfKind) {
    return getEnumName(cdfKind, cdfKindNames);
}

std::string cdfPrecisionName(CdfPrecision cdfPrecision) {
    return getEnumName(cdfPrecision, cdfPrecisionNames);
}


std::string weightClampingAlgorithmName(WeightClampingAlgorithm weightClampingAlgorithm) {
    return getEnumName(weightClampingAlgorithm, weightClampingAlgorithmNames);
}

std::string samplingMethodName(SamplingMethod samplingMethod) {
    return getEnumName(samplingMethod, samplingMethodNames);
}

std::string findBestNodeAlgorithmName(FindBestNodeAlgorithm findBestNodeAlgorithm) {
    return getEnumName(findBestNodeAlgorithm, findBestNodeAlgorithmNames);
}

// BVHStochSubsetAccel Utility Functions
inline uint32_t LeftShift3(uint32_t x) {
    CHECK_LE(x, (1u << 10));
    if (x == (1 << 10)) --x;
#ifdef PBRT_HAVE_BINARY_CONSTANTS
    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x | (x << 8)) & 0b00000011000000001111000000001111;
    // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x | (x << 4)) & 0b00000011000011000011000011000011;
    // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x | (x << 2)) & 0b00001001001001001001001001001001;
    // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
#else
    x = (x | (x << 16)) & 0x30000ff;
    // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x | (x << 8)) & 0x300f00f;
    // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x | (x << 4)) & 0x30c30c3;
    // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x | (x << 2)) & 0x9249249;
    // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
#endif  // PBRT_HAVE_BINARY_CONSTANTS
    return x;
}

inline uint32_t EncodeMorton3(const Vector3f &v) {
    CHECK_GE(v.x, 0);
    CHECK_GE(v.y, 0);
    CHECK_GE(v.z, 0);
    return (LeftShift3(v.z) << 2) | (LeftShift3(v.y) << 1) | LeftShift3(v.x);
}

float costSAH(BuildNode* root) {
    if (root->isLeaf()) return root->nPrimitives;

    auto l = root->children[0];
    auto r = root->children[1];
    return 1 + (
        l->bounds.SurfaceArea() * costSAH(l) +
        r->bounds.SurfaceArea() * costSAH(r)) /
        root->bounds.SurfaceArea();
}

float costSA(BuildNode *root) {
    float cost = 0.f;
    recurseNodes(root, [&](BuildNode* node) {
        if (node->isInterior())
            cost += node->bounds.SurfaceArea();
    });
    return cost;
}

float leafSA(BuildNode *root) {
    float cost = 0.f;
    recurseNodes(root, [&](BuildNode* node) {
        if (node->isLeaf())
            cost += node->bounds.SurfaceArea();
    });
    return cost;
}


float costSA2(BuildNode *root) {
    float cost = 0.f;
    recurseNodes(root, [&](BuildNode* node) {
        if (node->isInterior())
            cost += node->bounds.SurfaceArea();
        else
            cost += node->bounds.SurfaceArea() * node->nPrimitives;
    });
    return cost / root->bounds.SurfaceArea();
}

BuildNode* Build(const BuildParameters &params, BuildContext &ctx, std::vector<std::shared_ptr<Primitive>> &primitives) {
    if (primitives.empty()) return nullptr;
    // Build BVH from _primitives_

    // Initialize _primitiveInfo_ array for primitives
    size_t size = 0;
    for (size_t i = 0; i < primitives.size(); i++) {
        auto bounds = primitives[i]->WorldBound();
        if (bounds.pMin[0] == std::numeric_limits<float>::max()) continue;
        if (bounds.SurfaceArea() == 0) continue;
        float area = 1.f;
        if (const GeometricPrimitive *prim =
                dynamic_cast<GeometricPrimitive *>(primitives[i].get())) {
            area = prim->GetShape()->Area();
        } else {
            // TODO aggregate types. use bounding box area?
            area = 1.f;
        }
        ctx.bvhPrimitives[size++] =
            PrimitiveInfo(i, primitives[i]->WorldBound(), area);
    }

    // Build BVH tree for primitives using _primitiveInfo_
    BuildNode *root = ctx.arena.Alloc<BuildNode>();
    root->InitParent(nullptr);
    ctx.totalNodes++;
    BuildStochastic(params, ctx, primitives, root, size);

    if (false) {
        uint32_t expectedPrimOffset = 0;
        recurseNodes(root, [&](BuildNode *node) {
            if (node->isLeaf()) {
                if (node->firstPrimOffset != expectedPrimOffset)
                    abort();
                expectedPrimOffset += node->nPrimitives;
            }
        });
        if (expectedPrimOffset != size)
            abort();
    }

    ReorderPrimitives(ctx.bvhPrimitives, primitives, size);

    std::cout << "[bvhstochsubset] SAH Cost: " << costSAH(root) << std::endl;
    std::cout << "[bvhstochsubset] SA Cost: " << costSA(root) << std::endl;
    std::cout << "[bvhstochsubset] SA2 Cost: " << costSA2(root) << std::endl;

    return root;
}

// Main algorithm as detailed in the paper
void BuildStochastic(const BuildParameters &params,
                                     BuildContext &ctx,
                                     const std::vector<std::shared_ptr<Primitive>> &primitives,
                                     BuildNode *&root, int size) {
    if (size < params.minSetSize || params.subsetSizeFrac == 1.0) {
        RecursiveBuild(params.baseBuildParameters, ctx, root, ctx.bvhPrimitives, 0, size);
        return;
    }

    // Morton sorting of primitives
    {
        ProfilePhase p(Prof::AccelMortonSort);

        Bounds3f bounds;
        for (size_t i = 0; i < size; ++i)
            bounds = Union(bounds, ctx.bvhPrimitives[i].centroid());
 
        std::sort(ctx.bvhPrimitives.begin(),
                  ctx.bvhPrimitives.end(),
                  [&](const PrimitiveInfo &a,
                      const PrimitiveInfo &b) {
                      PBRT_CONSTEXPR int mortonBits = 10;
                      PBRT_CONSTEXPR int mortonScale = 1 << mortonBits;

                      // XXX offset [0,1] * 2^10 ? why?
                      Vector3f aV = bounds.Offset(a.centroid()) * mortonScale;
                      Vector3f bV = bounds.Offset(b.centroid()) * mortonScale;

                      return EncodeMorton3(aV) < EncodeMorton3(bV);
                  });
    }

    std::unordered_set<int> allPrimsSet;
    if (params.validateSubset) {
        for (int i = 0; i < size; i++) {
            allPrimsSet.insert(ctx.bvhPrimitives[i].primitiveNumber);
        }
    }

    // Subset sampling

    int mid;
    {
        ProfilePhase p(Prof::AccelSubsetCreation);
        switch (params.cdfPrecision) {
        case CdfPrecision::Single:
            mid = Subset<float>(params, ctx, primitives, size); break;
        case CdfPrecision::Double:
            mid = Subset<double>(params, ctx, primitives, size); break;
        default: LOG(FATAL) << "invalid cdfPrecision";
    }
    }

    if (params.validateSubset) {
        for (int i = 0; i < size; i++) {
            auto it = allPrimsSet.find(ctx.bvhPrimitivesScratch[i].primitiveNumber);
            CHECK(it != allPrimsSet.end());
            allPrimsSet.erase(it);
        }
    }

    // Subset BVH construction

    {
        ProfilePhase p(Prof::AccelSubsetRecursiveBuild);
        // build bvh from subset
        RecursiveBuild(params.subsetBuildParameters, ctx, root, ctx.bvhPrimitivesScratch, 0, mid);
    }

    // Insertion

    std::fill(ctx.nodeSelections.begin(), ctx.nodeSelections.end(), nullptr);

    recurseNodes(root, [&](BuildNode *node) {
        if (node->isLeaf())
            for (int i = 0; i < node->nPrimitives; i++) {
                const auto &prim = ctx.bvhPrimitivesScratch[node->firstPrimOffset + i];
                CHECK_LE(0, prim.primitiveNumber);
                CHECK_LT(prim.primitiveNumber, size);
                ctx.nodeSelections[prim.primitiveNumber] = node;
                //auto subsetIdx = ctx.closestSubsetPrimitive[prim.primitiveNumber];
                //ctx.subsetNodeSelections[subsetIdx] = node;
                // TODO this assertion will likely not always hold (overlapping primitives)
                if (params.validateBestNodeForSubsetPrimitives)
                    CHECK(FindBestNode(params, ctx, root, prim.primitiveNumber, 0, mid) == node);
            }
    });

    if (params.findBestNodeAlgorithm == FindBestNodeAlgorithm::LocalMortonSearch) {
        std::vector<bool> mask = ctx.subsetSelection;
        std::vector<int> backref;
        int j = 0;
        for (int i = 0; i < size; i++) {
            if (ctx.subsetSelection[i]) {
                ctx.subsetNodeSelections[j++] = ctx.nodeSelections[i];
                backref.push_back(i);
            }
        }

        ctx.subsetNodeSelectionsSize = j;

        {
            int j = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) j++;
                ctx.closestSubsetPrimitive[i] = j;
            }
        }
    }

    {
        ProfilePhase p(Prof::AccelFindBestNode);

        for (int i = 0; i < size; i++) {
            if (ctx.subsetSelection[i]) continue;
            auto node = FindBestNode(params, ctx, root, i, 0, mid);
            ctx.nodeSelections[i] = node;
        }

        // insert primitives into bvh
        recurseNodes(root, [&](BuildNode *node) {
            if (node->isLeaf()) {
                node->firstPrimOffset = 0;
            }
        });

        for (int i = 0; i < size; i++)
            ctx.nodeSelections[i]->firstPrimOffset++;
    }

    std::cout << "[bvhstochsubset] Cluster SA: " << leafSA(root) << std::endl;

    if (params.validateNonEmptyNodes) {
        recurseNodes(root, [&](BuildNode *node) {
            if (node->isLeaf())
                CHECK(node->firstPrimOffset > 0);
        });
    }

    int acc = 0;
    recurseNodes(root, [&](BuildNode *node) {
        if (node->isLeaf()) {
            int tmp = node->firstPrimOffset;
            node->firstPrimOffset = acc;
            acc += tmp;
            node->nPrimitives = 0;
        }
    });

    for (int i = 0; i < size; i++) {
        auto node = ctx.nodeSelections[i];
        auto offset = node->firstPrimOffset + node->nPrimitives++;
        ctx.bvhPrimitivesScratch[offset] = ctx.bvhPrimitives[i];
    }

    std::copy(ctx.bvhPrimitivesScratch.begin(),
              ctx.bvhPrimitivesScratch.end(),
              ctx.bvhPrimitives.begin());

    // Cluster BVHs construction

    {
        ProfilePhase p(Prof::AccelRecursiveBuild);

        // continue recursion
        recurseNodes(root, [&](BuildNode *node) {
            if (node->isLeaf()) {
                CHECK(node->nPrimitives > 0);
                RecursiveBuild(params.baseBuildParameters, ctx, node, ctx.bvhPrimitives,
                                node->firstPrimOffset, node->firstPrimOffset + node->nPrimitives);
            }
        });
    }

    // bounding boxes of all intermediate nodes are broken,
    // since they are based on just the stochastic subset they were constructed
    // with
    recurseNodesPost(root, [&](BuildNode *node) {
        if (node->isLeaf()) {
            node->bounds = Bounds3f();
            for (int i = 0; i < node->nPrimitives; i++) {
                node->bounds =
                    Union(node->bounds,
                          ctx.bvhPrimitives[node->firstPrimOffset + i].bounds);
            }
        } else {
            auto left = node->children[0];
            auto right = node->children[1];
            node->bounds = Union(left->bounds, right->bounds);
            node->nPrimitives = 0;
        }
    });
}

// Subset sampling
template<typename T>
int Subset(const BuildParameters &params, BuildContext &ctx,
    const std::vector<std::shared_ptr<Primitive>> &primitives,
    int size) {

    if (params.subsetSizeFrac == 1.0) {
        for (int i = 0; i < size; i++) {
            auto prim = ctx.bvhPrimitives[i];
            prim.primitiveNumber = i;
            ctx.bvhPrimitivesScratch[i] = prim;
        }
        std::fill(ctx.subsetSelection.begin(), ctx.subsetSelection.end(), true);
        return size;
    }

    std::vector<T> cdf(size);

    // Primitive weights & CDF
    T sum = 0.f;
    for (int i = 0; i < size; i++) {
        auto bounds = ctx.bvhPrimitives[i].bounds;
        float weight = bounds.Diagonal().Length();
        CHECK(weight > 0.f);
        cdf[i] = weight;
        sum += weight;
    }

    /*
    {
        std::ofstream ofs("weights.bin");
        ofs.write(reinterpret_cast<const char*>(cdf.data()), sizeof(cdf[0]) * size);
    }
    */

    // Algorithm 1 
    if (params.weightClampingAlgorithm == WeightClampingAlgorithm::Binned) {
        const float BIN_BASE = std::sqrt(2.f);
        const int BIN_OFFSET = 32;
        const int BIN_COUNT = 64;

        int bin_counts[BIN_COUNT];

        for (int i = 0; i < BIN_COUNT; i++)
            bin_counts[i] = 0;

        for (int i = 0; i < size; i++) {
            auto w = cdf[i];
            if (w == 0) continue;
            size_t bin;
            if (w == std::numeric_limits<T>::infinity())
                bin = BIN_COUNT - 1;
            else
                bin = std::min<float>(std::max<float>(BIN_OFFSET + std::floor(std::log(w)/std::log(BIN_BASE)), 0.f), BIN_COUNT - 1.f);

            bin_counts[bin]++;
        }

        T s = 1.f / (size * params.subsetSizeFrac);
        switch(params.samplingMethod) {
            case SamplingMethod::Equidistant: break;
            case SamplingMethod::SobolFill:
            case SamplingMethod::Sobol: s *= 2; break;
            default: LOG(FATAL) << "invalid samplingMethod";
        }
        
        // Compensate uniformity (Eq. 6)
        s = (s - params.uniformCdfFrac / size) / (1 - params.uniformCdfFrac);

        T uSum = 0;
        T cSum = size;
        T clamp;

        for (int i = 0; i < BIN_COUNT; i++) {
            if (i == BIN_COUNT - 1) {
                clamp = std::numeric_limits<T>::infinity();
                break;
            }
            clamp = pow(BIN_BASE, i-BIN_OFFSET+1);
            if (clamp / (uSum + clamp * cSum) >= s) break;
            uSum += clamp * bin_counts[i];
            cSum -= bin_counts[i];
        }

        sum = 0.f;
        for (int i = 0; i < size; i++) {
            T weight = std::min(cdf[i], clamp);
            sum += weight;
            cdf[i] = weight;
        }
    }

    T pfix = 0.f;
    for (int i = 0; i < size; i++) {
        // Uniformity
        pfix += cdf[i] * (1 - params.uniformCdfFrac) + sum * params.uniformCdfFrac / (T)(size);
        cdf[i] = pfix;
        ctx.cdf[i] = (double)pfix;
    }

    sum = pfix;

    // Subset
    int subsetSize = (int)std::round(size * params.subsetSizeFrac);

    std::fill(ctx.subsetSelection.begin(), ctx.subsetSelection.end(), false);
    //std::cout << "\n\nbuild:" << params.subsetSizeMin << " " << params.subsetSizeMax << " " << params.subsetSizeFrac << "\n\n";
    std::cout << "FINAL SUBSET: " << subsetSize << " from: " << size << std::endl;

    int samples = subsetSize;
    for (int i = 0; i < samples; i++) {
        T rnd;
        switch (params.samplingMethod) {
            case SamplingMethod::Equidistant: rnd = T(i) / samples; break;
            case SamplingMethod::Sobol:
            case SamplingMethod::SobolFill: rnd = SobolSampleDouble(i,0,0); break;
            default: LOG(FATAL) << "invalid samplingMethod";
        }

        auto it = std::upper_bound(cdf.begin(), cdf.end() - 1, rnd * sum);

        auto j = it - cdf.begin();
        auto selection = ctx.subsetSelection[j];
        if (params.samplingMethod == SamplingMethod::SobolFill && selection)
            samples++;
        selection = true;
    }

    ctx.subsetSize = 0;
    for (int i = 0; i < size; i++) {
        if (!ctx.subsetSelection[i]) continue;

        auto prim = ctx.bvhPrimitives[i];
        prim.primitiveNumber = i; // Backreference to original primitive
        ctx.bvhPrimitivesScratch[ctx.subsetSize++] =  prim;
    }

    return ctx.subsetSize;
}

template
int Subset<float>(const BuildParameters &params, BuildContext &ctx,
    const std::vector<std::shared_ptr<Primitive>> &primitives,
    int size);

template
int Subset<double>(const BuildParameters &params, BuildContext &ctx,
    const std::vector<std::shared_ptr<Primitive>> &primitives,
    int size);

// Algorithm 2
BuildNode *FindBestNode(
    const BuildParameters &params, BuildContext &ctx, BuildNode *root,
    const int primitiveIdx, int begin, int end) {
    const auto &primitive = ctx.bvhPrimitives[primitiveIdx];

    // Cost: Increase of SAH
    auto metric = [&](const BuildNode *a) {
        CHECK(a->isLeaf());

        auto nodeBounds = a->bounds;
        auto oldSA = nodeBounds.SurfaceArea();
        auto newSA = Union(nodeBounds, primitive.bounds).SurfaceArea();
        float cost = newSA * (a->nPrimitives + 1) - oldSA * a->nPrimitives;

        const BuildNode* node = a;
        while(node->parent != nullptr) {
            auto nodeBounds = node->bounds;
            auto oldSA = nodeBounds.SurfaceArea();
            auto newSA = Union(nodeBounds, primitive.bounds).SurfaceArea();
            cost += newSA - oldSA;
            node = node->parent;
        };
        return cost;
    };

    switch (params.findBestNodeAlgorithm) {
        case FindBestNodeAlgorithm::BruteForce: {
            float mincost = FLT_MAX; 
            float maxcost = -FLT_MAX; 
            BuildNode *bestnode = nullptr;

            // insert primitives into bvh
            recurseNodes(root, [&](BuildNode *node) {
                if (node->isLeaf()) {
                    float currmin = metric(node);
                    if(currmin < mincost)
                    {
                        mincost = currmin;
                        bestnode = node;
                    }
                }
            });

            return bestnode;
        }
        // Morton window search
        case FindBestNodeAlgorithm::LocalMortonSearch: {
            float mincost = FLT_MAX; 
            BuildNode *bestnode = nullptr;

            auto subsetPrimitiveIdx = ctx.closestSubsetPrimitive[primitiveIdx];
            auto begin = std::max(subsetPrimitiveIdx - params.localMortonSearchWindow, 0);
            auto end = std::min(subsetPrimitiveIdx + params.localMortonSearchWindow, ctx.subsetNodeSelectionsSize);
            for (int i = begin; i < end; i++) {
                auto node = ctx.subsetNodeSelections[i];
                float currmin = metric(node);
                if(currmin < mincost)
                {
                    mincost = currmin;
                    bestnode = node;
                }
            }

            return bestnode;
        }
        default: LOG(FATAL) << "invalid findBestNodeAlgorithm";
    }
}

// Recursive binned SAH builder
void RecursiveBuild(const RecursiveBuildParameters &params, BuildContext &ctx,
                           BuildNode *node,
                           std::vector<PrimitiveInfo> &primitiveInfo,
                           int start, int end) {
    CHECK_NE(start, end);
    // Compute bounds of all primitives in BVH node
    Bounds3f bounds;
    for (int i = start; i < end; ++i)
        bounds = Union(bounds, primitiveInfo[i].bounds);
    int nPrimitives = end - start;
    if (nPrimitives == 1) {
        // Create leaf _BuildNode_
        node->InitLeaf(start, end - start, bounds);
        return;
    }

    // Compute bound of primitive centroids, choose split dimension
    // _dim_
    Bounds3f centroidBounds;
    for (int i = start; i < end; ++i)
        centroidBounds = Union(centroidBounds, primitiveInfo[i].centroid());
    int dim = centroidBounds.MaximumExtent();

    // Partition primitives into two sets and build children
    int mid = (start + end) / 2;
    if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
        // Create leaf _BuildNode_
        node->InitLeaf(start, end - start, bounds);
        return;
    }

    // Partition primitives based on binned sah
    {
        // Allocate _BucketInfo_ for SAH partition buckets
        for (int dim = 0; dim < 3; dim++)
        for (int i = 0; i < params.binCount; i++)
            ctx.bins[dim][i] = BVHBucketInfo();

        bool skipDimension[3];
        for (int dim = 0; dim < 3; dim++)
            skipDimension[dim] = centroidBounds.pMin[dim] == centroidBounds.pMax[dim];

        // Initialize _BucketInfo_ for SAH partition buckets
        for (int i = start; i < end; ++i) {
            auto offset = centroidBounds.Offset(primitiveInfo[i].centroid());
            for (int dim = 0; dim < 3; dim++) {
                if (skipDimension[dim]) continue;
                int b = params.binCount * offset[dim];
                if (b == params.binCount) b = params.binCount - 1;
                CHECK_GE(b, 0);
                CHECK_LT(b, params.binCount);
                ctx.bins[dim][b].count += 1;
                //ctx.bins[dim][b].count++;
                ctx.bins[dim][b].bounds =
                    Union(ctx.bins[dim][b].bounds, primitiveInfo[i].bounds);
            }
        }

        // Compute costs for splitting after each bucket
        for (int dim = 0; dim < 3; dim++) {
            if (skipDimension[dim]) continue;
            for (int i = 0; i < params.binCount - 1; ++i) {
                Bounds3f b0, b1;
                float count0 = 0, count1 = 0;
                for (int j = 0; j <= i; ++j) {
                    b0 = Union(b0, ctx.bins[dim][j].bounds);
                    count0 += ctx.bins[dim][j].count;
                }
                for (int j = i + 1; j < params.binCount; ++j) {
                    b1 = Union(b1, ctx.bins[dim][j].bounds);
                    count1 += ctx.bins[dim][j].count;
                }
                if (count0 == 0 || count1 == 0) {
                    ctx.costs[dim][i] = FLT_MAX;   
                    continue;
                }
                ctx.costs[dim][i] = 1 + (count0 * b0.SurfaceArea() +
                                    count1 * b1.SurfaceArea()) /
                                    bounds.SurfaceArea();
            }
        }

        // Find bucket to split at that minimizes SAH metric
        Float minCost = FLT_MAX;
        int minCostDim = 0;
        int minCostSplitBucket = 0;
        for (int dim = 0; dim < 3; dim++) {
            if (skipDimension[dim]) continue;
            for (int i = 0; i < params.binCount - 1; ++i) {
                if (ctx.costs[dim][i] < minCost) {
                    minCost = ctx.costs[dim][i];
                    minCostDim = dim;
                    minCostSplitBucket = i;
                }
            }
        }

        CHECK_NE(minCost, FLT_MAX);

        // Either create leaf or split primitives at selected
        // SAH bucket
        Float leafCost = nPrimitives;
        if (nPrimitives > params.maxPrimsInNode ||
            (params.ignoreLeafCost ? false : minCost < leafCost)) {
            PrimitiveInfo *pmid = std::partition(
                &primitiveInfo[start], &primitiveInfo[end - 1] + 1,
                [=](const PrimitiveInfo &pi) {
                    int b = params.binCount * centroidBounds.Offset(
                                            pi.centroid())[minCostDim];
                    if (b == params.binCount) b = params.binCount - 1;
                    CHECK_GE(b, 0);
                    CHECK_LT(b, params.binCount);
                    return b <= minCostSplitBucket;
                });
            mid = pmid - &primitiveInfo[0];
        } else {
            // Create leaf _BuildNode_
            node->InitLeaf(start, end - start, bounds);
            return;
        }
        //}
    }

    BuildNode *left = ctx.arena.Alloc<BuildNode>();
    left->InitParent(node);
    ctx.totalNodes++;
    RecursiveBuild(params, ctx, left, primitiveInfo, start, mid);

    BuildNode *right = ctx.arena.Alloc<BuildNode>();
    right->InitParent(node);
    ctx.totalNodes++;
    RecursiveBuild(params, ctx, right, primitiveInfo, mid, end);

    node->InitInterior(dim, left, right);
    return;
}

void ReorderPrimitives(const std::vector<PrimitiveInfo> &bvhPrimitives,
                       std::vector<std::shared_ptr<Primitive>> &primitives,
                       const size_t size) {
    std::vector<std::shared_ptr<Primitive>> newPrimitives(primitives.size());

    for (int i = 0; i < size; i++)
        newPrimitives[i] = primitives[bvhPrimitives[i].primitiveNumber];

    primitives = std::move(newPrimitives);
}

bool contains(Bounds3f out, Bounds3f in) {
    return
        out.pMin.x <= in.pMin.x && in.pMax.x <= out.pMax.x && 
        out.pMin.y <= in.pMin.y && in.pMax.y <= out.pMax.y && 
        out.pMin.z <= in.pMin.z && in.pMax.z <= out.pMax.z;
}

}

using namespace bvhstochsubset;

// BVHStochSubsetAccel Method Definitions
BVHStochSubsetAccel::BVHStochSubsetAccel(
    const BuildParameters &params, std::vector<std::shared_ptr<Primitive>> p)
    : primitives(std::move(p)) {
    ProfilePhase _(Prof::AccelConstruction);

    BuildContext ctx(1024 * 1024, primitives.size(), std::max(params.subsetBuildParameters.binCount, params.baseBuildParameters.binCount));
    BuildNode* root = Build(params, ctx, primitives);
    if (root == nullptr) return;

    ctx.bvhPrimitives.resize(0);
    LOG(INFO) << StringPrintf(
        "BVH created with %d nodes for %d "
        "primitives (%.2f MB), arena allocated %.2f MB",
        ctx.totalNodes, (int)primitives.size(),
        float(ctx.totalNodes * sizeof(LinearBVHNode)) / (1024.f * 1024.f),
        float(ctx.arena.TotalAllocated()) / (1024.f * 1024.f));

    // Compute representation of depth-first traversal of BVH tree
    treeBytes += ctx.totalNodes * sizeof(LinearBVHNode) + sizeof(*this) +
                 primitives.size() * sizeof(primitives[0]);
    nodes = AllocAligned<LinearBVHNode>(ctx.totalNodes);
    int offset = 0;
    flattenBVHTree(root, &offset);
    //CHECK_EQ(ctx.totalNodes, offset);
}

Bounds3f BVHStochSubsetAccel::WorldBound() const {
    return nodes ? nodes[0].bounds : Bounds3f();
}

int BVHStochSubsetAccel::flattenBVHTree(BuildNode *node, int *offset) {
    LinearBVHNode *linearNode = &nodes[*offset];
    linearNode->bounds = node->bounds;
    int myOffset = (*offset)++;
    if (node->isLeaf()) {
        CHECK(!node->children[0] && !node->children[1]);
        CHECK_LT(node->nPrimitives, 65536);
        linearNode->primitivesOffset = node->firstPrimOffset;
        linearNode->nPrimitives = node->nPrimitives;
        for (int i = 0; i < linearNode->nPrimitives; i++)
            CHECK(contains(linearNode->bounds, primitives[linearNode->primitivesOffset + i]->WorldBound()));
    } else {
        // Create interior flattened BVH node
        linearNode->axis = node->splitAxis;
        linearNode->nPrimitives = 0;
        flattenBVHTree(node->children[0], offset);
        linearNode->secondChildOffset =
            flattenBVHTree(node->children[1], offset);
        CHECK(contains(linearNode->bounds, nodes[myOffset + 1].bounds));
        CHECK(contains(linearNode->bounds, nodes[linearNode->secondChildOffset].bounds));
    }
    return myOffset;
}

BVHStochSubsetAccel::~BVHStochSubsetAccel() { FreeAligned(nodes); }

bool BVHStochSubsetAccel::Intersect(const Ray &ray,
                                    SurfaceInteraction *isect) const {
    if (!nodes) return false;
    ProfilePhase p(Prof::AccelIntersect);
    bool hit = false;
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = {invDir.x < 0, invDir.y < 0, invDir.z < 0};
    // Follow ray through BVH nodes to find primitive intersections
    int toVisitOffset = 0, currentNodeIndex = 0;
    int nodesToVisit[64];
    while (true) {
        const LinearBVHNode *node = &nodes[currentNodeIndex];
        // test number of intersections
        nodeIntersect++;
        getNodeIntersectionsCounter()++;
        // Check ray against BVH node
        if (node->bounds.IntersectP(ray, invDir, dirIsNeg)) {
            if (node->nPrimitives > 0) {
                // Intersect ray with primitives in leaf BVH node
                // XXX intersect multiple times and does not check min distance?
                for (int i = 0; i < node->nPrimitives; ++i) {
                    primIntersect++;
                    getPrimIntersectionsCounter()++;
                    if (primitives[node->primitivesOffset + i]->Intersect(
                            ray, isect))
                        hit = true;
                }
                if (toVisitOffset == 0) break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];
            } else {
                // Put far BVH node on _nodesToVisit_ stack, advance to near
                // node
                if (dirIsNeg[node->axis]) {
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->secondChildOffset;
                } else {
                    nodesToVisit[toVisitOffset++] = node->secondChildOffset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        } else {
            if (toVisitOffset == 0) break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }
    return hit;
}

bool BVHStochSubsetAccel::IntersectP(const Ray &ray) const {
    if (!nodes) return false;
    ProfilePhase p(Prof::AccelIntersectP);
    Vector3f invDir(1.f / ray.d.x, 1.f / ray.d.y, 1.f / ray.d.z);
    int dirIsNeg[3] = {invDir.x < 0, invDir.y < 0, invDir.z < 0};
    int nodesToVisit[64];
    int toVisitOffset = 0, currentNodeIndex = 0;
    while (true) {
        // test number of intersections
        nodeIntersect++;
        getNodeIntersectionsCounter()++;
        const LinearBVHNode *node = &nodes[currentNodeIndex];
        if (node->bounds.IntersectP(ray, invDir, dirIsNeg)) {
            // Process BVH node _node_ for traversal
            if (node->nPrimitives > 0) {
                for (int i = 0; i < node->nPrimitives; ++i) {
                    primIntersect++;
                    getPrimIntersectionsCounter()++;
                    if (primitives[node->primitivesOffset + i]->IntersectP(
                            ray)) {
                        return true;
                    }
                }
                if (toVisitOffset == 0) break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];
            } else {
                if (dirIsNeg[node->axis]) {
                    /// second child first
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->secondChildOffset;
                } else {
                    nodesToVisit[toVisitOffset++] = node->secondChildOffset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        } else {
            if (toVisitOffset == 0) break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }
    return false;
}

template<typename T, typename F>
void parseParameter(const ParamSet &ps, std::string key, T& parameter, F nameFun) {
    std::string parameterName = ps.FindOneString(key, nameFun(parameter));
    for (int i = 0; i < (int)T::Max; i++) {
        if (parameterName == nameFun((T)i)) {
            parameter = (T)i;
            return;
        }
    }
    Warning("parameter \"%s\" unknown. Using default.",
        parameterName.c_str());
}

std::shared_ptr<BVHStochSubsetAccel> CreateBVHStochSubsetAccelerator(
    std::vector<std::shared_ptr<Primitive>> prims, const ParamSet &ps) {
    BuildParameters params;

    params.minSetSize = ps.FindOneInt("minsetsize", params.minSetSize);
    if (params.minSetSize < 0) {
        params.minSetSize = 0;
        Warning("minsetsize < 0. Using minsetsize = 0.");
    }

    params.subsetSizeFrac =
        ps.FindOneFloat("subsetsizefrac", params.subsetSizeFrac);
    if (params.subsetSizeFrac < 0.f || 1.f < params.subsetSizeFrac) {
        params.subsetSizeFrac =
            std::min(std::max(params.subsetSizeFrac, 0.f), 1.f);
        Warning("subsetsizefrac is not in [0, 1]. Using subsetsizefrac=%f.",
                params.subsetSizeFrac);
    }

    parseParameter(ps, "cdfkind", params.cdfKind, cdfKindName);
    parseParameter(ps, "cdfprecision", params.cdfPrecision, cdfPrecisionName);
    parseParameter(ps, "weightclampingalgorithm", params.weightClampingAlgorithm, weightClampingAlgorithmName);
    params.uniformCdfFrac = ps.FindOneFloat("uniformcdffrac", params.uniformCdfFrac);
    parseParameter(ps, "samplingmethod", params.samplingMethod, samplingMethodName);
    parseParameter(ps, "findbestnodealgorithm", params.findBestNodeAlgorithm, findBestNodeAlgorithmName);

    int localMortonSearchWindow = ps.FindOneInt("localmortonsearchwindow", params.localMortonSearchWindow);
    if (localMortonSearchWindow <= 0) {
        localMortonSearchWindow = params.localMortonSearchWindow;
        Warning("localmortonsearchwindow is negative. Using localmortonsearchwindow=%i.",
                params.localMortonSearchWindow);
    }
    params.localMortonSearchWindow = localMortonSearchWindow;

    if (params.subsetSizeFrac < 0.f || 1.f < params.subsetSizeFrac) {
        params.subsetSizeFrac =
            std::min(std::max(params.subsetSizeFrac, 0.f), 1.f);
    }


    auto parseRecursiveBuildParameters = [&](RecursiveBuildParameters &params, std::string prefix) {
        params.binCount = ps.FindOneInt(prefix + ".binCount", params.binCount);

        params.maxPrimsInNode = ps.FindOneInt(prefix + ".maxprimsinnode", params.maxPrimsInNode);
        if (params.maxPrimsInNode < 1) {
            params.maxPrimsInNode = 1;
            Warning("maxprimsinnode < 1. Using maxprimsinnode = 1.");
        }

        params.ignoreLeafCost = ps.FindOneBool(prefix + ".ignoreleafcost", params.ignoreLeafCost);
    };

    parseRecursiveBuildParameters(params.subsetBuildParameters, "subset");
    parseRecursiveBuildParameters(params.baseBuildParameters, "base");

    params.validateSubset = ps.FindOneBool("validateSubset", params.validateSubset);
    params.validateBestNodeForSubsetPrimitives = ps.FindOneBool("validateBestNodeForSubsetPrimitives", params.validateBestNodeForSubsetPrimitives);
    params.validateNonEmptyNodes = ps.FindOneBool("validateNonEmptyNodes", params.validateNonEmptyNodes);

    params.printParameters(4);

    return std::shared_ptr<BVHStochSubsetAccel>(
        new BVHStochSubsetAccel(params, std::move(prims)));
}

}  // namespace pbrt
