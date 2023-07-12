; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 2
; RUN: opt -aa-pipeline=basic-aa -passes=attributor -attributor-manifest-internal -S < %s | FileCheck %s --check-prefixes=CHECK,TUNIT

declare float @llvm.fabs.f32(float)
declare float @llvm.copysign.f32(float, float)
declare i1 @llvm.is.fpclass.f32(float, i32 immarg)

; Arithmetic fence is to workaround an attributor bug where select
; seems to be special cased when returned, such that
; computeKnownFPClass is never called on it.

declare float @llvm.arithmetic.fence.f32(float)

define float @ret_select_nnan_flag(i1 %cond, float %arg0, float %arg1) {
; CHECK-LABEL: define float @ret_select_nnan_flag
; CHECK-SAME: (i1 [[COND:%.*]], float [[ARG0:%.*]], float [[ARG1:%.*]]) #[[ATTR1:[0-9]+]] {
; CHECK-NEXT:    [[SELECT:%.*]] = select nnan i1 [[COND]], float [[ARG0]], float [[ARG1]]
; CHECK-NEXT:    ret float [[SELECT]]
;
  %select = select nnan i1 %cond, float %arg0, float %arg1
  ret float %select
}

define float @ret_select_ninf_flag(i1 %cond, float %arg0, float %arg1) {
; CHECK-LABEL: define float @ret_select_ninf_flag
; CHECK-SAME: (i1 [[COND:%.*]], float [[ARG0:%.*]], float [[ARG1:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[SELECT:%.*]] = select ninf i1 [[COND]], float [[ARG0]], float [[ARG1]]
; CHECK-NEXT:    ret float [[SELECT]]
;
  %select = select ninf i1 %cond, float %arg0, float %arg1
  ret float %select
}

define float @ret_select_nnan_ninf_flag(i1 %cond, float %arg0, float %arg1) {
; CHECK-LABEL: define float @ret_select_nnan_ninf_flag
; CHECK-SAME: (i1 [[COND:%.*]], float [[ARG0:%.*]], float [[ARG1:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[SELECT:%.*]] = select nnan ninf i1 [[COND]], float [[ARG0]], float [[ARG1]]
; CHECK-NEXT:    ret float [[SELECT]]
;
  %select = select nnan ninf i1 %cond, float %arg0, float %arg1
  ret float %select
}

define float @ret_fence_select_nnan_flag(i1 %cond, float %arg0, float %arg1) {
; CHECK-LABEL: define nofpclass(nan) float @ret_fence_select_nnan_flag
; CHECK-SAME: (i1 [[COND:%.*]], float [[ARG0:%.*]], float [[ARG1:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[SELECT:%.*]] = select nnan i1 [[COND]], float [[ARG0]], float [[ARG1]]
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(nan) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2:[0-9]+]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %select = select nnan i1 %cond, float %arg0, float %arg1
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

; TODO: Remove all the fences beyond here when attributor bug is fixed.

define float @ret_select_nonan__noinf_nonan(i1 %cond, float nofpclass(nan) %arg0, float nofpclass(nan inf) %arg1) {
; CHECK-LABEL: define nofpclass(nan) float @ret_select_nonan__noinf_nonan
; CHECK-SAME: (i1 [[COND:%.*]], float nofpclass(nan) [[ARG0:%.*]], float nofpclass(nan inf) [[ARG1:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[COND]], float [[ARG0]], float [[ARG1]]
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(nan) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %select = select i1 %cond, float %arg0, float %arg1
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

; Clamp nan to 0 pattern
define float @ret_select_clamp_nan_to_zero_uno(float %arg) {
; CHECK-LABEL: define nofpclass(nan) float @ret_select_clamp_nan_to_zero_uno
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[IS_NAN:%.*]] = fcmp uno float [[ARG]], 0.000000e+00
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_NAN]], float 0.000000e+00, float [[ARG]]
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(nan) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %is.nan = fcmp uno float %arg, 0.0
  %select = select i1 %is.nan, float 0.0, float %arg
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @ret_select_clamp_nan_to_zero_ord(float %arg) {
; CHECK-LABEL: define nofpclass(nan) float @ret_select_clamp_nan_to_zero_ord
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[NOT_NAN:%.*]] = fcmp ord float [[ARG]], 0.000000e+00
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[NOT_NAN]], float [[ARG]], float 0.000000e+00
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(nan) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %not.nan = fcmp ord float %arg, 0.0
  %select = select i1 %not.nan, float %arg, float 0.0
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @ret_select_clamp_onlynans(float %arg) {
; CHECK-LABEL: define nofpclass(inf zero sub norm) float @ret_select_clamp_onlynans
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[NOT_NAN:%.*]] = fcmp ord float [[ARG]], 0.000000e+00
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[NOT_NAN]], float 0x7FF8000000000000, float [[ARG]]
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(inf zero sub norm) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %not.nan = fcmp ord float %arg, 0.0
  %select = select i1 %not.nan, float 0x7FF8000000000000, float %arg
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @clamp_nonfinite_to_normal_olt(float %arg) {
; CHECK-LABEL: define nofpclass(nan inf) float @clamp_nonfinite_to_normal_olt
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[FABS:%.*]] = call float @llvm.fabs.f32(float [[ARG]]) #[[ATTR2]]
; CHECK-NEXT:    [[IS_FINITE:%.*]] = fcmp olt float [[FABS]], 0x7FF0000000000000
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_FINITE]], float [[ARG]], float 1.024000e+03
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(nan inf) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %fabs = call float @llvm.fabs.f32(float %arg)
  %is.finite = fcmp olt float %fabs, 0x7FF0000000000000
  %select = select i1 %is.finite, float %arg, float 1024.0
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @clamp_eq_inf_to_pnormal(float %arg) {
; CHECK-LABEL: define nofpclass(inf) float @clamp_eq_inf_to_pnormal
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[FABS:%.*]] = call float @llvm.fabs.f32(float [[ARG]]) #[[ATTR2]]
; CHECK-NEXT:    [[IS_INF:%.*]] = fcmp oeq float [[FABS]], 0x7FF0000000000000
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_INF]], float 1.024000e+03, float [[ARG]]
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(inf) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %fabs = call float @llvm.fabs.f32(float %arg)
  %is.inf = fcmp oeq float %fabs, 0x7FF0000000000000
  %select = select i1 %is.inf, float 1024.0, float %arg
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @clamp_eq_pinf_to_pnormal(float %arg) {
; CHECK-LABEL: define nofpclass(pinf) float @clamp_eq_pinf_to_pnormal
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[IS_INF:%.*]] = fcmp oeq float [[ARG]], 0x7FF0000000000000
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_INF]], float 1.024000e+03, float [[ARG]]
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(pinf) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %is.inf = fcmp oeq float %arg, 0x7FF0000000000000
  %select = select i1 %is.inf, float 1024.0, float %arg
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @clamp_eq_ninf_to_negnormal(float %arg) {
; CHECK-LABEL: define nofpclass(ninf) float @clamp_eq_ninf_to_negnormal
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[IS_INF:%.*]] = fcmp oeq float [[ARG]], 0xFFF0000000000000
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_INF]], float -1.024000e+03, float [[ARG]]
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(ninf) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %is.inf = fcmp oeq float %arg, 0xFFF0000000000000
  %select = select i1 %is.inf, float -1024.0, float %arg
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @clamp_eq_inf_to_nan(float %arg) {
; CHECK-LABEL: define nofpclass(inf) float @clamp_eq_inf_to_nan
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[FABS:%.*]] = call float @llvm.fabs.f32(float [[ARG]]) #[[ATTR2]]
; CHECK-NEXT:    [[IS_INF:%.*]] = fcmp oeq float [[FABS]], 0x7FF0000000000000
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_INF]], float 0x7FF8000000000000, float [[ARG]]
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(inf) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %fabs = call float @llvm.fabs.f32(float %arg)
  %is.inf = fcmp oeq float %fabs, 0x7FF0000000000000
  %select = select i1 %is.inf, float 0x7FF8000000000000, float %arg
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @ret_select_clamp_nan_to_zero_uno_returned_different_arg(float %arg0, float %arg1) {
; CHECK-LABEL: define float @ret_select_clamp_nan_to_zero_uno_returned_different_arg
; CHECK-SAME: (float [[ARG0:%.*]], float [[ARG1:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[IS_NAN:%.*]] = fcmp uno float [[ARG0]], 0.000000e+00
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_NAN]], float 0.000000e+00, float [[ARG1]]
; CHECK-NEXT:    [[FENCE:%.*]] = call float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %is.nan = fcmp uno float %arg0, 0.0
  %select = select i1 %is.nan, float 0.0, float %arg1
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @isfinite_select_fabs_val_0(float %arg) {
; CHECK-LABEL: define nofpclass(nan inf nzero nsub nnorm) float @isfinite_select_fabs_val_0
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[FABS:%.*]] = call float @llvm.fabs.f32(float [[ARG]]) #[[ATTR2]]
; CHECK-NEXT:    [[IS_FINITE:%.*]] = fcmp olt float [[FABS]], 0x7FF0000000000000
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_FINITE]], float [[FABS]], float 1.024000e+03
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(nan inf nzero nsub nnorm) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %fabs = call float @llvm.fabs.f32(float %arg)
  %is.finite = fcmp olt float %fabs, 0x7FF0000000000000
  %select = select i1 %is.finite, float %fabs, float 1024.0
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @isfinite_select_fabs_val_1(float %arg) {
; CHECK-LABEL: define nofpclass(nan inf nzero nsub nnorm) float @isfinite_select_fabs_val_1
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[FABS:%.*]] = call float @llvm.fabs.f32(float [[ARG]]) #[[ATTR2]]
; CHECK-NEXT:    [[NOT_IS_FINITE:%.*]] = fcmp uge float [[FABS]], 0x3810000000000000
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[NOT_IS_FINITE]], float 1.024000e+03, float [[FABS]]
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(nan inf nzero nsub nnorm) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %fabs = call float @llvm.fabs.f32(float %arg)
  %not.is.finite = fcmp uge float %fabs, 0x3810000000000000
  %select = select i1 %not.is.finite, float 1024.0, float %fabs
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @clamp_denormal_to_poszero(float %arg) {
; CHECK-LABEL: define nofpclass(nzero sub) float @clamp_denormal_to_poszero
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[FABS:%.*]] = call float @llvm.fabs.f32(float [[ARG]]) #[[ATTR2]]
; CHECK-NEXT:    [[IS_DENORM_OR_ZERO:%.*]] = fcmp olt float [[FABS]], 0x3810000000000000
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_DENORM_OR_ZERO]], float 0.000000e+00, float [[ARG]]
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(nzero sub) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %fabs = call float @llvm.fabs.f32(float %arg)
  %is.denorm.or.zero = fcmp olt float %fabs, 0x3810000000000000
  %select = select i1 %is.denorm.or.zero, float 0.0, float %arg
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @clamp_denormal_to_negzero(float %arg) {
; CHECK-LABEL: define nofpclass(pzero sub) float @clamp_denormal_to_negzero
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[FABS:%.*]] = call float @llvm.fabs.f32(float [[ARG]]) #[[ATTR2]]
; CHECK-NEXT:    [[IS_DENORM_OR_ZERO:%.*]] = fcmp olt float [[FABS]], 0x3810000000000000
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_DENORM_OR_ZERO]], float -0.000000e+00, float [[ARG]]
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(pzero sub) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %fabs = call float @llvm.fabs.f32(float %arg)
  %is.denorm.or.zero = fcmp olt float %fabs, 0x3810000000000000
  %select = select i1 %is.denorm.or.zero, float -0.0, float %arg
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @clamp_denormal_to_zero_copysign(float %arg) {
; CHECK-LABEL: define nofpclass(sub) float @clamp_denormal_to_zero_copysign
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[FABS:%.*]] = call float @llvm.fabs.f32(float [[ARG]]) #[[ATTR2]]
; CHECK-NEXT:    [[IS_DENORM_OR_ZERO:%.*]] = fcmp olt float [[FABS]], 0x3810000000000000
; CHECK-NEXT:    [[ZERO:%.*]] = call float @llvm.copysign.f32(float noundef 0.000000e+00, float [[ARG]]) #[[ATTR2]]
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_DENORM_OR_ZERO]], float [[ZERO]], float [[ARG]]
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(sub) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %fabs = call float @llvm.fabs.f32(float %arg)
  %is.denorm.or.zero = fcmp olt float %fabs, 0x3810000000000000
  %zero = call float @llvm.copysign.f32(float 0.0, float %arg)
  %select = select i1 %is.denorm.or.zero, float %zero, float %arg
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @clamp_only_denormal_or_zero(float %arg) {
; CHECK-LABEL: define nofpclass(nan inf norm) float @clamp_only_denormal_or_zero
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[FABS:%.*]] = call float @llvm.fabs.f32(float [[ARG]]) #[[ATTR2]]
; CHECK-NEXT:    [[IS_DENORM_OR_ZERO:%.*]] = fcmp olt float [[FABS]], 0x3810000000000000
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_DENORM_OR_ZERO]], float [[ARG]], float 0.000000e+00
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(nan inf norm) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %fabs = call float @llvm.fabs.f32(float %arg)
  %is.denorm.or.zero = fcmp olt float %fabs, 0x3810000000000000
  %select = select i1 %is.denorm.or.zero, float %arg, float 0.0
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @clamp_inf_to_fabs(float %arg) {
; CHECK-LABEL: define float @clamp_inf_to_fabs
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[FABS:%.*]] = call float @llvm.fabs.f32(float [[ARG]]) #[[ATTR2]]
; CHECK-NEXT:    [[IS_INF:%.*]] = fcmp oeq float [[FABS]], 0x7FF0000000000000
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_INF]], float [[FABS]], float [[ARG]]
; CHECK-NEXT:    [[FENCE:%.*]] = call float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %fabs = call float @llvm.fabs.f32(float %arg)
  %is.inf = fcmp oeq float %fabs, 0x7FF0000000000000
  %select = select i1 %is.inf, float %fabs, float %arg
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @not_clamp_inf_to_fabs(float %arg) {
; CHECK-LABEL: define float @not_clamp_inf_to_fabs
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[FABS:%.*]] = call float @llvm.fabs.f32(float [[ARG]]) #[[ATTR2]]
; CHECK-NEXT:    [[IS_INF:%.*]] = fcmp oeq float [[FABS]], 0x7FF0000000000000
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_INF]], float [[ARG]], float [[FABS]]
; CHECK-NEXT:    [[FENCE:%.*]] = call float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %fabs = call float @llvm.fabs.f32(float %arg)
  %is.inf = fcmp oeq float %fabs, 0x7FF0000000000000
  %select = select i1 %is.inf, float %arg, float %fabs
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @clamp_zero_to_inf(float %arg) {
; CHECK-LABEL: define nofpclass(zero) float @clamp_zero_to_inf
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[IS_ZERO:%.*]] = fcmp oeq float [[ARG]], 0.000000e+00
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_ZERO]], float 0x7FF0000000000000, float [[ARG]]
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(zero) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %is.zero = fcmp oeq float %arg, 0.0
  %select = select i1 %is.zero, float 0x7FF0000000000000, float %arg
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @clamp_zero_to_only_inf(float %arg) {
; CHECK-LABEL: define nofpclass(nan ninf sub norm) float @clamp_zero_to_only_inf
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[IS_ZERO:%.*]] = fcmp oeq float [[ARG]], 0.000000e+00
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_ZERO]], float [[ARG]], float 0x7FF0000000000000
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(nan ninf sub norm) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %is.zero = fcmp oeq float %arg, 0.0
  %select = select i1 %is.zero, float %arg, float 0x7FF0000000000000
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @clamp_is_class_subnormal_or_inf_to_nan(float %arg) {
; CHECK-LABEL: define nofpclass(inf sub) float @clamp_is_class_subnormal_or_inf_to_nan
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[IS_SUBNORMAL_OR_INF:%.*]] = call i1 @llvm.is.fpclass.f32(float [[ARG]], i32 noundef 660) #[[ATTR2]]
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_SUBNORMAL_OR_INF]], float 0x7FF8000000000000, float [[ARG]]
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(inf sub) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %is.subnormal.or.inf = call i1 @llvm.is.fpclass.f32(float %arg, i32 660)
  %select = select i1 %is.subnormal.or.inf, float 0x7FF8000000000000, float %arg
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @clamp_is_class_subnormal_or_inf_to_nan_swap(float %arg) {
; CHECK-LABEL: define nofpclass(inf sub) float @clamp_is_class_subnormal_or_inf_to_nan_swap
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[NOT_IS_SUBNORMAL_OR_INF:%.*]] = call i1 @llvm.is.fpclass.f32(float [[ARG]], i32 noundef 363) #[[ATTR2]]
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[NOT_IS_SUBNORMAL_OR_INF]], float [[ARG]], float 0x7FF8000000000000
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(inf sub) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %not.is.subnormal.or.inf = call i1 @llvm.is.fpclass.f32(float %arg, i32 363)
  %select = select i1 %not.is.subnormal.or.inf, float %arg, float 0x7FF8000000000000
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @ret_select_clamp_nan_to_zero_fpclass(float %arg) {
; CHECK-LABEL: define nofpclass(nan) float @ret_select_clamp_nan_to_zero_fpclass
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[IS_NAN:%.*]] = call i1 @llvm.is.fpclass.f32(float [[ARG]], i32 noundef 3) #[[ATTR2]]
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_NAN]], float 0.000000e+00, float [[ARG]]
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(nan) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %is.nan = call i1 @llvm.is.fpclass.f32(float %arg, i32 3)
  %select = select i1 %is.nan, float 0.0, float %arg
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @ret_select_clamp_snan_to_zero_fpclass(float %arg) {
; CHECK-LABEL: define nofpclass(snan) float @ret_select_clamp_snan_to_zero_fpclass
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[IS_NAN:%.*]] = call i1 @llvm.is.fpclass.f32(float [[ARG]], i32 noundef 1) #[[ATTR2]]
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_NAN]], float 0.000000e+00, float [[ARG]]
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(snan) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %is.nan = call i1 @llvm.is.fpclass.f32(float %arg, i32 1)
  %select = select i1 %is.nan, float 0.0, float %arg
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @ret_select_clamp_qnan_to_zero_fpclass(float %arg) {
; CHECK-LABEL: define nofpclass(qnan) float @ret_select_clamp_qnan_to_zero_fpclass
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[IS_NAN:%.*]] = call i1 @llvm.is.fpclass.f32(float [[ARG]], i32 noundef 2) #[[ATTR2]]
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_NAN]], float 0.000000e+00, float [[ARG]]
; CHECK-NEXT:    [[FENCE:%.*]] = call nofpclass(qnan) float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %is.nan = call i1 @llvm.is.fpclass.f32(float %arg, i32 2)
  %select = select i1 %is.nan, float 0.0, float %arg
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @ret_select_clamp_nan_to_zero_fpclass_other_val(float %arg0, float %arg1) {
; CHECK-LABEL: define float @ret_select_clamp_nan_to_zero_fpclass_other_val
; CHECK-SAME: (float [[ARG0:%.*]], float [[ARG1:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[IS_NAN:%.*]] = call i1 @llvm.is.fpclass.f32(float [[ARG0]], i32 noundef 3) #[[ATTR2]]
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_NAN]], float 0.000000e+00, float [[ARG1]]
; CHECK-NEXT:    [[FENCE:%.*]] = call float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %is.nan = call i1 @llvm.is.fpclass.f32(float %arg0, i32 3)
  %select = select i1 %is.nan, float 0.0, float %arg1
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @clamp_is_denorm_or_zero_to_fneg(float %arg) {
; CHECK-LABEL: define float @clamp_is_denorm_or_zero_to_fneg
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[FABS:%.*]] = call float @llvm.fabs.f32(float [[ARG]]) #[[ATTR2]]
; CHECK-NEXT:    [[IS_DENORM_OR_ZERO:%.*]] = fcmp olt float [[FABS]], 0x3810000000000000
; CHECK-NEXT:    [[NEG_ARG:%.*]] = fneg float [[ARG]]
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_DENORM_OR_ZERO]], float [[NEG_ARG]], float [[ARG]]
; CHECK-NEXT:    [[FENCE:%.*]] = call float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %fabs = call float @llvm.fabs.f32(float %arg)
  %is.denorm.or.zero = fcmp olt float %fabs, 0x3810000000000000
  %neg.arg = fneg float %arg
  %select = select i1 %is.denorm.or.zero, float %neg.arg, float %arg
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @select_is_denorm_or_zero_to_fneg_or_fabs(float %arg) {
; CHECK-LABEL: define float @select_is_denorm_or_zero_to_fneg_or_fabs
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[FABS:%.*]] = call float @llvm.fabs.f32(float [[ARG]]) #[[ATTR2]]
; CHECK-NEXT:    [[IS_DENORM_OR_ZERO:%.*]] = fcmp olt float [[FABS]], 0x3810000000000000
; CHECK-NEXT:    [[NEG_ARG:%.*]] = fneg float [[ARG]]
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_DENORM_OR_ZERO]], float [[NEG_ARG]], float [[FABS]]
; CHECK-NEXT:    [[FENCE:%.*]] = call float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %fabs = call float @llvm.fabs.f32(float %arg)
  %is.denorm.or.zero = fcmp olt float %fabs, 0x3810000000000000
  %neg.arg = fneg float %arg
  %select = select i1 %is.denorm.or.zero, float %neg.arg, float %fabs
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

define float @select_is_denorm_or_zero_to_fabs_or_fneg(float %arg) {
; CHECK-LABEL: define float @select_is_denorm_or_zero_to_fabs_or_fneg
; CHECK-SAME: (float [[ARG:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:    [[FABS:%.*]] = call float @llvm.fabs.f32(float [[ARG]]) #[[ATTR2]]
; CHECK-NEXT:    [[IS_DENORM_OR_ZERO:%.*]] = fcmp olt float [[FABS]], 0x3810000000000000
; CHECK-NEXT:    [[NEG_ARG:%.*]] = fneg float [[ARG]]
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[IS_DENORM_OR_ZERO]], float [[FABS]], float [[NEG_ARG]]
; CHECK-NEXT:    [[FENCE:%.*]] = call float @llvm.arithmetic.fence.f32(float [[SELECT]]) #[[ATTR2]]
; CHECK-NEXT:    ret float [[FENCE]]
;
  %fabs = call float @llvm.fabs.f32(float %arg)
  %is.denorm.or.zero = fcmp olt float %fabs, 0x3810000000000000
  %neg.arg = fneg float %arg
  %select = select i1 %is.denorm.or.zero, float %fabs, float %neg.arg
  %fence = call float @llvm.arithmetic.fence.f32(float %select)
  ret float %fence
}

;; NOTE: These prefixes are unused and the list is autogenerated. Do not add tests below this line:
; TUNIT: {{.*}}