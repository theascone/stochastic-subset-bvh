
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

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_ACCELERATORS_BVH_STOCH_SUBSET_H
#define PBRT_ACCELERATORS_BVH_STOCH_SUBSET_H

// accelerators/bvh.h*
#include <atomic>
#include <array>
#include <unordered_set>

#include "pbrt.h"
#include "primitive.h"

namespace pbrt {

namespace bvhstochsubset {
    enum class CdfKind { Uniform = 0, Diagonal, Area, Max };
    enum class CdfPrecision { Single = 0, Double, Max };
    enum class WeightClampingAlgorithm { Disabled = 0, Binned, Max };
    enum class SamplingMethod { Equidistant = 0, Sobol, SobolFill, Max };
    enum class FindBestNodeAlgorithm { BruteForce = 0, LocalMortonSearch, Max };

    static const char *cdfKindNames[] = {
        "uniform",
        "diagonal",
        "area",
    };
    static const char *cdfPrecisionNames[] = {
        "single",
        "double",
    };
    static const char *weightClampingAlgorithmNames[] = {
        "disabled",
        "binned",
    };
    static const char *samplingMethodNames[] = {
        "equidistant",
        "sobol",
        "sobolfill",
    };
    static const char *findBestNodeAlgorithmNames[] = {
        "bruteforce",
        "localmortonsearch",
    };

    struct RecursiveBuildParameters {
        int binCount;
        int maxPrimsInNode;
        bool ignoreLeafCost;

        void printParameters(int indent);
    };

    struct BuildParameters {
        int minSetSize = 10000;
        float subsetSizeFrac = 0.5; // Subset size relative to total number primitives
        CdfKind cdfKind = CdfKind::Diagonal; // What weight to use for sampling
        // Precision of the cdf. we use double here,
        // since our cdf accumulation is just a linear scan and therefore more sensitive to numerical errors
        CdfPrecision cdfPrecision = CdfPrecision::Double; 
        // Enable/Disable binned weight clamping
        WeightClampingAlgorithm weightClampingAlgorithm = WeightClampingAlgorithm::Binned;
        // Added uniformity to sampling
        float uniformCdfFrac = 0.0;
        // Sampling method for drawing out primitives according to the CDF
        SamplingMethod samplingMethod = SamplingMethod::Equidistant;
        FindBestNodeAlgorithm findBestNodeAlgorithm = FindBestNodeAlgorithm::LocalMortonSearch;
        int localMortonSearchWindow = 10;
        RecursiveBuildParameters subsetBuildParameters = {12, 10, true};
        RecursiveBuildParameters baseBuildParameters = {12, 10, false};

        bool validateSubset = false;
        bool validateBestNodeForSubsetPrimitives = false;
        bool validateNonEmptyNodes = false;

        void printParameters(int indent);
    };

    struct PrimitiveInfo {
        PrimitiveInfo() {}
        PrimitiveInfo(size_t primitiveNumber, Bounds3f bounds,
                              float area) {
            this->primitiveNumber = primitiveNumber;
            this->bounds = bounds;
            this->area = area;
        }
        size_t primitiveNumber;
        Bounds3f bounds;
        float area;
        //float cost;
        Point3f centroid() const { return .5f * bounds.pMin + .5f * bounds.pMax; }
    };

    struct BuildNode {
        void InitParent(BuildNode* parent) {
            this->parent = parent;
        }

        // BuildNode Public Methods
        void InitLeaf(int first, int n, const Bounds3f &b) {
            firstPrimOffset = first;
            nPrimitives = n;
            bounds = b;
            children[0] = children[1] = nullptr;
        }
        void InitInterior(int axis, BuildNode *c0, BuildNode *c1) {
            children[0] = c0;
            children[1] = c1;
            bounds = Union(c0->bounds, c1->bounds);
            splitAxis = axis;
            nPrimitives = 0;
        }
        bool isInterior() const { return children[0]; };
        bool isLeaf() const { return !isInterior(); };
        Point3f centroid() const { return (bounds.pMin + bounds.pMax) / 2; };
        BuildNode* parent;
        Bounds3f bounds;
        std::array<BuildNode*, 2> children;
        int splitAxis, firstPrimOffset, nPrimitives;
    };

    struct BVHBucketInfo {
        float count = 0;
        Bounds3f bounds;
    };

    // Auxiliary Buffers needed for the construction
    struct BuildContext {
        MemoryArena arena;
        std::vector<PrimitiveInfo> bvhPrimitives;
        std::vector<PrimitiveInfo> bvhPrimitivesScratch;
        std::vector<int> closestSubsetPrimitive;
        std::vector<double> cdf;
        std::vector<BuildNode*> nodeSelections;
        std::vector<BuildNode*> subsetNodeSelections;
        int subsetNodeSelectionsSize = 0;
        std::vector<bool> subsetSelection;
        std::vector<BVHBucketInfo> bins[3];
        std::vector<float> costs[3];
        int totalNodes = 0;
        int subsetSize = 0;

        BuildContext(size_t arenaSize, size_t numPrims, size_t numBins)
            : arena(arenaSize),
              bvhPrimitives(numPrims),
              bvhPrimitivesScratch(numPrims),
              closestSubsetPrimitive(numPrims),
              cdf(numPrims),
              nodeSelections(numPrims),
              subsetNodeSelections(numPrims),
              subsetSelection(numPrims),
              bins{
                  std::vector<BVHBucketInfo>(numBins),
                  std::vector<BVHBucketInfo>(numBins),
                  std::vector<BVHBucketInfo>(numBins)},
              costs{
                  std::vector<float>(numBins),
                  std::vector<float>(numBins),
                  std::vector<float>(numBins)} { }
    };

    struct LinearBVHNode {
        Bounds3f bounds;
        union {
            int primitivesOffset;   // leaf
            int secondChildOffset;  // interior
        };
        uint16_t nPrimitives;  // 0 -> interior node
        uint8_t axis;          // interior node: xyz
        uint8_t pad[1];        // ensure 32 byte total size
    };

    template<class E, class A>
    const char* getEnumName(E value, const A &nameArray) {
        if ((int)value < (int)E::Max) {
            return nameArray[(int)value];
        }
        throw std::runtime_error("invalid enum value!");
    }

    std::string cdfKindName(CdfKind cdfKind);
    std::string cdfPrecisionName(CdfPrecision cdfPrecision);
    std::string weightClampingAlgorithmName(WeightClampingAlgorithm weightClampingAlgorithm);
    std::string samplingMethodName(SamplingMethod samplingMethod);
    std::string findBestNodeAlgorithmName(FindBestNodeAlgorithm findBestNodeAlgorithm);

    uint32_t LeftShift3(uint32_t x);
    uint32_t EncodeMorton3(const Vector3f &v);

    template <typename T>
    static void recurseNodes(BuildNode *root, T t) {
        t(root);
        if (root->isInterior()) {
            recurseNodes(root->children[0], t);
            recurseNodes(root->children[1], t);
        }
    }

    template <typename T>
    static void recurseNodesPost(BuildNode *root, T t) {
        if (root->isInterior()) {
            recurseNodesPost(root->children[0], t);
            recurseNodesPost(root->children[1], t);
        }
        t(root);
    }

    float costSAH(BuildNode* root);
    float costSA(BuildNode *root);
    float costSA2(BuildNode *root);

    BuildNode* Build(const BuildParameters &params, BuildContext &ctx, std::vector<std::shared_ptr<Primitive>> &primitives);
    void BuildStochastic(const BuildParameters &params,
                                  BuildContext &ctx, const std::vector<std::shared_ptr<Primitive>> &primitives, BuildNode *&root,
                                  int size);
    template<typename T>
    int Subset(const BuildParameters &params, BuildContext &ctx, const std::vector<std::shared_ptr<Primitive>> &primitives, int size);

    BuildNode *FindBestNode(
        const BuildParameters &params, BuildContext &ctx, BuildNode *root,
        const int primitiveIdx, int begin, int end);
    void RecursiveBuild(
        const RecursiveBuildParameters &params, BuildContext &ctx,
        BuildNode *node,
        std::vector<PrimitiveInfo> &primitiveInfo,
        int start, int end);
    void ReorderPrimitives(const std::vector<PrimitiveInfo> &bvhPrimitives,
                    std::vector<std::shared_ptr<Primitive>> &primitives,
                    const size_t size);

    void DumpPrimitiveInfo(std::string filePath, const std::vector<PrimitiveInfo> &bvhPrimitives);
    std::vector<PrimitiveInfo> LoadPrimitiveInfo(std::string filePath);
}

// BVHAccel Declarations
class BVHStochSubsetAccel : public Aggregate {
  public:
    // BVHAccel Public Types

    // BVHAccel Public Methods
    BVHStochSubsetAccel(const bvhstochsubset::BuildParameters &params,
                        std::vector<std::shared_ptr<Primitive>> p);
    Bounds3f WorldBound() const;
    ~BVHStochSubsetAccel();
    bool Intersect(const Ray &ray, SurfaceInteraction *isect) const;
    bool IntersectP(const Ray &ray) const;

  private:
    // BVHAccel Private Methods
    int flattenBVHTree(bvhstochsubset::BuildNode *node, int *offset);

    // BVHAccel Private Data
    std::vector<std::shared_ptr<Primitive>> primitives;
    bvhstochsubset::LinearBVHNode *nodes = nullptr;
};

std::shared_ptr<BVHStochSubsetAccel> CreateBVHStochSubsetAccelerator(
    std::vector<std::shared_ptr<Primitive>> prims, const ParamSet &ps);

}  // namespace pbrt

#endif  // PBRT_ACCELERATORS_BVH_STOCH_SUBSET_H
