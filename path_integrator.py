import os
import enoki as ek
import mitsuba
import numpy

# Set the desired mitsuba variant
mitsuba.set_variant('gpu_rgb')

from mitsuba.core import Float, Vector3f,Spectrum, Thread, xml, RayDifferential3f
from mitsuba.core.xml import load_file
from mitsuba.render import (BSDF, BSDFContext, BSDFFlags,
                            DirectionSample3f, Emitter, ImageBlock,
                            SamplingIntegrator, has_flag,
                            register_integrator)


def mis_weight(pdf_a, pdf_b):
    pdf_a *= pdf_a
    pdf_b *= pdf_b
    return ek.select(pdf_a > 0.0, pdf_a / (pdf_a + pdf_b), Float(0.0))


def integrator(scene, sampler, rays, medium, active=True):
    si = scene.ray_intersect(rays)
    si_orig = si
    active = si.is_valid() & active
    emitter_vis = si.emitter(scene, active)
    eta = Float(1.0)
    emission_weight = Float(1.0)
    throughput = Spectrum(1.0)
    result = Spectrum(0.0)
    
    for depths in range(200):
        # Visible emitters
        result += ek.select(active, emission_weight*throughput*Emitter.eval_vec(emitter_vis, si, active), Vector3f(0.0))
        active = si.is_valid() & active

        # Russian Roulette
        if (depths > 1):
            q = ek.min(ek.hmax(throughput) * ek.sqr(eta), Float(0.95))
            active &= sampler.next_1d(active) < q
            throughput *= ek.rcp(q)

        if not ek.any(active):
            print(depths)
            break

        ctx = BSDFContext()
        bsdf = si.bsdf(rays)

        # Emitter sampling
        sample_emitter = active & has_flag(BSDF.flags_vec(bsdf), BSDFFlags.Smooth)
        ds, emitter_val = scene.sample_emitter_direction(si, sampler.next_2d(sample_emitter), True, sample_emitter)
        active_e = sample_emitter & ek.neq(ds.pdf, 0.0)
        wo = si.to_local(ds.d)
        bsdf_val = BSDF.eval_vec(bsdf, ctx, si, wo, active_e)
        bsdf_pdf = BSDF.pdf_vec(bsdf, ctx, si, wo, active_e)
        mis = ek.select(ds.delta, Float(1), mis_weight(ds.pdf, bsdf_pdf))
        result += ek.select(active_e, emitter_val * bsdf_val * mis * throughput, Vector3f(0))

        # BSDF sampling
        active_b = active
        bs, bsdf_val = BSDF.sample_vec(bsdf, ctx, si, sampler.next_1d(active), sampler.next_2d(active), active_b)
        throughput = throughput * bsdf_val
        eta = eta * bs.eta
        
        rays = RayDifferential3f(si.spawn_ray(si.to_world(bs.wo)))
        si_bsdf = scene.ray_intersect(rays, active_b)
        emitter = si_bsdf.emitter(scene, active_b)
        active_b &= ek.neq(emitter, 0)
        # emitter_val = Emitter.eval_vec(emitter, si_bsdf, active_b)
        delta = has_flag(bs.sampled_type, BSDFFlags.Delta)
        ds = DirectionSample3f(si_bsdf, si)
        ds.object = emitter
        emitter_pdf = ek.select(delta, Float(0), scene.pdf_emitter_direction(si, ds, active_b))
        emission_weight = mis_weight(bs.pdf,emitter_pdf)
        si = si_bsdf
        # result += ek.select(active_b, bsdf_val * emitter_val * mis_weight(bs.pdf, emitter_pdf), Vector3f(0))
    return result, si_orig.is_valid(), ek.select(si_orig.is_valid(), si_orig.t, Float(0.0))


class MyDirectIntegrator(SamplingIntegrator):
    def __init__(self, props):
        SamplingIntegrator.__init__(self, props)

    def sample(self, scene, sampler, ray, medium, active):
        result, is_valid, depth = integrator(scene, sampler, ray, medium, active)
        return result, is_valid, [depth]

    def aov_names(self):
        return ["depth.Y"]

    def to_string(self):
        return "MyPathIntegrator[]"


# Register our integrator such that the XML file loader can instantiate it when loading a scene
register_integrator("mydirectintegrator", lambda props: MyDirectIntegrator(props))

# Load an XML file which specifies "mydirectintegrator" as the scene's integrator
filename = './cbox/cbox.xml'
Thread.thread().file_resolver().append(os.path.dirname(filename))
scene = load_file(filename)

scene.integrator().render(scene, scene.sensors()[0])

film = scene.sensors()[0].film()
film.set_destination_file('my-path-integrator-rrtest.exr')
film.develop()
