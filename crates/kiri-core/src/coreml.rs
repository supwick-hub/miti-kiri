// Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
//! Core ML backend (macOS): the compiled model on the CPU and Neural Engine, via objc2.
#![allow(deprecated)]
use crate::backend::{Backend, Format};
use crate::spec::{f16_bits_to_f32, Geometry};
use anyhow::{anyhow, bail, Result};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::AnyThread;
use objc2_core_ml::*;
use objc2_foundation::*;
use std::ffi::c_void;
use std::path::Path;
use std::ptr::NonNull;

pub struct CoreMl {
    geom: Geometry,
    model: Retained<MLModel>,
    in_name: Retained<NSString>,
    out_name: Retained<NSString>,
    shape: Retained<NSArray<NSNumber>>,
    strides: Retained<NSArray<NSNumber>>,
}

fn nums(v: &[isize]) -> Retained<NSArray<NSNumber>> {
    let items: Vec<Retained<NSNumber>> = v.iter().map(|&x| NSNumber::new_isize(x)).collect();
    NSArray::from_retained_slice(&items)
}

/// Every (window, stem, channel, bin) row of an output of `b` windows, in memory order.
fn rows(b: isize) -> impl Iterator<Item = (isize, isize, isize, isize)> {
    (0..b).flat_map(|n| (0..4).flat_map(move |s| (0..4).flat_map(move |c| (0..2049).map(move |k| (n, s, c, k)))))
}

impl CoreMl {
    /// `geom`: the model's input shape (`Geometry::from_model`); Core ML rejects a call of any other shape.
    pub fn load(path: &Path, geom: Geometry) -> Result<Self> {
        let (b, t) = (geom.batch as isize, geom.frames as isize);
        unsafe {
            let url = NSURL::fileURLWithPath_isDirectory(&NSString::from_str(&path.to_string_lossy()), true);
            let cfg = MLModelConfiguration::new();
            // The GPU by default: it loads in under a second where the Neural Engine took 3-8 min to
            // prepare, and it matches CPU fp32 to 46-61 dB where the Neural Engine's fp16 is 19-38
            // (2026-10-06). KIRI_UNITS=ane|all|cpu|gpu to compare.
            let units = match std::env::var("KIRI_UNITS").as_deref() {
                Ok("ane") => MLComputeUnits::CPUAndNeuralEngine,
                Ok("all") => MLComputeUnits::All,
                Ok("cpu") => MLComputeUnits::CPUOnly,
                _ => MLComputeUnits::CPUAndGPU,
            };
            cfg.setComputeUnits(units);
            let model = MLModel::modelWithContentsOfURL_configuration_error(&url, &cfg)
                .map_err(|e| anyhow!("loading {}: {}", path.display(), e.localizedDescription()))?;
            Ok(Self {
                geom,
                model,
                in_name: NSString::from_str("mix_spec"),
                out_name: NSString::from_str("stems_spec"),
                shape: nums(&[b, 4, 2049, t]),
                strides: nums(&[4 * 2049 * t, 2049 * t, t, 1]),
            })
        }
    }
}

impl Backend for CoreMl {
    fn geometry(&self) -> Geometry {
        self.geom
    }

    fn separate_batch(&mut self, input: &[f32], output: &mut [f32]) -> Result<Format> {
        let g = self.geom;
        let (b, t) = (g.batch as isize, g.frames as isize);
        let total = g.batch * g.out_len();
        assert_eq!(input.len(), g.batch * g.in_len());
        assert_eq!(output.len(), total);
        autoreleasepool(|_| unsafe {
            // zero-copy: Core ML reads our buffer in place; it outlives the call
            let ptr = NonNull::new(input.as_ptr() as *mut c_void).unwrap();
            let arr = MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
                MLMultiArray::alloc(),
                ptr,
                &self.shape,
                MLMultiArrayDataType::Float32,
                &self.strides,
                None,
            )
            .map_err(|e| anyhow!("input array: {}", e.localizedDescription()))?;
            let fv = MLFeatureValue::featureValueWithMultiArray(&arr);
            let dict: Retained<NSDictionary<NSString, AnyObject>> =
                NSDictionary::from_retained_objects(&[&*self.in_name], &[Retained::into_super(Retained::into_super(fv))]);
            let provider = MLDictionaryFeatureProvider::initWithDictionary_error(MLDictionaryFeatureProvider::alloc(), &dict)
                .map_err(|e| anyhow!("features: {}", e.localizedDescription()))?;
            let out = self
                .model
                .predictionFromFeatures_error(ProtocolObject::from_ref(&*provider))
                .map_err(|e| anyhow!("predict: {}", e.localizedDescription()))?;
            let val = out.featureValueForName(&self.out_name).ok_or_else(|| anyhow!("no stems_spec output"))?;
            let arr = val.multiArrayValue().ok_or_else(|| anyhow!("stems_spec is not an array"))?;
            let dt = arr.dataType();
            let half = dt == MLMultiArrayDataType::Float16;
            if !half && dt != MLMultiArrayDataType::Float32 {
                bail!("stems_spec is neither float16 nor float32");
            }
            let shape: Vec<isize> = (0..arr.shape().count()).map(|i| arr.shape().objectAtIndex(i).as_isize()).collect();
            if shape != [b, 4, 4, 2049, t] {
                bail!("unexpected output shape {shape:?}");
            }
            let st: Vec<isize> = (0..arr.strides().count()).map(|i| arr.strides().objectAtIndex(i).as_isize()).collect();
            let p = arr.dataPointer().as_ptr();
            let contiguous = st == [4 * 4 * 2049 * t, 4 * 2049 * t, 2049 * t, t, 1];
            let mut fmt = Format::F32;
            if half {
                if contiguous {
                    std::ptr::copy_nonoverlapping(p as *const u16, output.as_mut_ptr() as *mut u16, total);
                    fmt = Format::F16;
                } else {
                    for (n, s, c, k) in rows(b) {
                        let src = (p as *const u16).offset(n * st[0] + s * st[1] + c * st[2] + k * st[3]);
                        let o = &mut output[((((n * 4 + s) * 4 + c) * 2049 + k) * t) as usize..][..t as usize];
                        for i in 0..t {
                            o[i as usize] = f16_bits_to_f32(*src.offset(i * st[4]));
                        }
                    }
                }
            } else if contiguous {
                std::ptr::copy_nonoverlapping(p as *const f32, output.as_mut_ptr(), total);
            } else {
                for (n, s, c, k) in rows(b) {
                    let src = (p as *const f32).offset(n * st[0] + s * st[1] + c * st[2] + k * st[3]);
                    let o = &mut output[((((n * 4 + s) * 4 + c) * 2049 + k) * t) as usize..][..t as usize];
                    for i in 0..t {
                        o[i as usize] = *src.offset(i * st[4]);
                    }
                }
            }
            Ok(fmt)
        })
    }
}

