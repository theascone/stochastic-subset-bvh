
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

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

// integrators/stat.cpp*
#include "integrators/stat.h"
#include "sampling.h"
#include "interaction.h"
#include "paramset.h"
#include "camera.h"
#include "film.h"
#include "scene.h"
#include "core/counters.h"

namespace pbrt {

// StatIntegrator Method Definitions
StatIntegrator::StatIntegrator(std::shared_ptr<const Camera> camera,
                           std::shared_ptr<Sampler> sampler,
                           const Bounds2i &pixelBounds)
    : SamplerIntegrator(camera, sampler, pixelBounds) {
}

Spectrum StatIntegrator::Li(const RayDifferential &r, const Scene &scene,
                          Sampler &sampler, MemoryArena &arena,
                          int depth) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    RayDifferential ray(r);

    // Intersect _ray_ with scene and store intersection in _isect_
    SurfaceInteraction isect;

    getNodeIntersectionsCounter() = 0;
    getPrimIntersectionsCounter() = 0;
    scene.Intersect(ray, &isect);

    return RGBSpectrum((float)getNodeIntersectionsCounter() / 1024.f);
    float rgb[3] = {(float)getNodeIntersectionsCounter(), (float)getPrimIntersectionsCounter(), 0.f};
    RGBSpectrum spectrum;
    return RGBSpectrum::FromRGB(rgb);
}

StatIntegrator *CreateStatIntegrator(const ParamSet &params,
                                 std::shared_ptr<Sampler> sampler,
                                 std::shared_ptr<const Camera> camera) {
    int np;
    const int *pb = params.FindInt("pixelbounds", &np);
    Bounds2i pixelBounds = camera->film->GetSampleBounds();
    if (pb) {
        if (np != 4)
            Error("Expected four values for \"pixelbounds\" parameter. Got %d.",
                  np);
        else {
            pixelBounds = Intersect(pixelBounds,
                                    Bounds2i{{pb[0], pb[2]}, {pb[1], pb[3]}});
            if (pixelBounds.Area() == 0)
                Error("Degenerate \"pixelbounds\" specified.");
        }
    }
    return new StatIntegrator(camera, sampler, pixelBounds);
}

}  // namespace pbrt
