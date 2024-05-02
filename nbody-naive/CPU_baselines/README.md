I could not find any good sequential benchmarks on the internet,
so resorted to writing some myself. They are not as fast as SaC though,
so should be improved. The difference seems to be in the tight loop.
Fortran and C get 13 Gflops/s on one core, SaC 16 Gflops/s.
Except for some register names, Fortran and C are the same (which makes sense
of course as both use the same optimiser). The difference in efficiency seems
to come from the summation conditional on whether n is 0 or not. It is also weird
that there is no broadcast for positions[j]
I am not convinced any of these are optimal, so maybe we should hand-vectorize the
loop using intrinsics for a fair baseline.

# Fortran

vmovupd (%rax,%rdx,1),%ymm7
vsubpd %ymm12,%ymm7,%ymm3
vmovupd (%rbx,%rdx,1),%ymm7
vsubpd %ymm11,%ymm7,%ymm2
vmovupd (%r11,%rdx,1),%ymm7
vmulpd %ymm2,%ymm2,%ymm0
vfmadd231pd %ymm3,%ymm3,%ymm0
vsubpd %ymm13,%ymm7,%ymm1
vxorpd %xmm7,%xmm7,%xmm7
vfmadd231pd %ymm1,%ymm1,%ymm0
vsqrtpd %ymm0,%ymm4
vmulpd %ymm4,%ymm0,%ymm0
vcmpneqpd %ymm7,%ymm0,%ymm4
vdivpd %ymm0,%ymm14,%ymm0
vmaskmovpd (%rdi,%rdx,1),%ymm4,%ymm7
add    $0x20,%rdx
vmulpd %ymm7,%ymm2,%ymm2
vmulpd %ymm7,%ymm1,%ymm1
vmulpd %ymm7,%ymm3,%ymm3
vmulpd %ymm0,%ymm2,%ymm2
vmulpd %ymm0,%ymm1,%ymm1
vmulpd %ymm0,%ymm3,%ymm3
vandpd %ymm4,%ymm2,%ymm2
vandpd %ymm4,%ymm1,%ymm1
vaddpd %ymm2,%ymm5,%ymm5
vaddpd %ymm1,%ymm6,%ymm6
vandpd %ymm4,%ymm3,%ymm3
vaddpd %ymm3,%ymm8,%ymm8
cmp    %rdx,%r9
jne    1390 <__nbody_mod_MOD_accelerateall+0x120>

# C

vmovupd (%rdx,%rax,1),%ymm7
vsubpd %ymm12,%ymm7,%ymm3
vmovupd (%rcx,%rax,1),%ymm7
vsubpd %ymm11,%ymm7,%ymm2
vmovupd (%rsi,%rax,1),%ymm7
vmulpd %ymm2,%ymm2,%ymm0
vfmadd231pd %ymm3,%ymm3,%ymm0
vsubpd %ymm13,%ymm7,%ymm1
vxorpd %xmm7,%xmm7,%xmm7
vfmadd231pd %ymm1,%ymm1,%ymm0
vsqrtpd %ymm0,%ymm4
vmulpd %ymm4,%ymm0,%ymm0
vcmpneqpd %ymm7,%ymm0,%ymm4
vdivpd %ymm0,%ymm14,%ymm0
vmaskmovpd (%r8,%rax,1),%ymm4,%ymm7
add    $0x20,%rax
vmulpd %ymm7,%ymm2,%ymm2
vmulpd %ymm7,%ymm1,%ymm1
vmulpd %ymm7,%ymm3,%ymm3
vmulpd %ymm0,%ymm2,%ymm2
vmulpd %ymm0,%ymm1,%ymm1
vmulpd %ymm0,%ymm3,%ymm3
vandpd %ymm4,%ymm2,%ymm2
vandpd %ymm4,%ymm1,%ymm1
vaddpd %ymm2,%ymm5,%ymm5
vaddpd %ymm1,%ymm6,%ymm6
vandpd %ymm4,%ymm3,%ymm3
vaddpd %ymm3,%ymm8,%ymm8
cmp    %r10,%rax
jne    1840 <accelerateAll+0xb0>

# SaC

vmovupd (%rcx,%rax,1),%ymm1
vmovupd (%r12,%rax,1),%ymm2
vmovupd (%r8,%rax,1),%ymm12
vsubpd %ymm9,%ymm1,%ymm10
vsubpd %ymm8,%ymm2,%ymm2
vmovupd (%rbx,%rax,1),%ymm1
add    $0x20,%rax
vmulpd %ymm2,%ymm2,%ymm0
vfmadd231pd %ymm10,%ymm10,%ymm0
vsubpd %ymm7,%ymm1,%ymm1
vfmadd231pd %ymm1,%ymm1,%ymm0
vsqrtpd %ymm0,%ymm11
vmulpd %ymm11,%ymm0,%ymm11
vcmpeqpd %ymm3,%ymm0,%ymm0
vdivpd %ymm11,%ymm12,%ymm11
vmulpd %ymm11,%ymm2,%ymm2
vmulpd %ymm11,%ymm1,%ymm1
vmulpd %ymm11,%ymm10,%ymm10
vblendvpd %ymm0,%ymm3,%ymm2,%ymm2
vblendvpd %ymm0,%ymm3,%ymm1,%ymm1
vaddpd %ymm2,%ymm5,%ymm5
vaddpd %ymm1,%ymm6,%ymm6
vblendvpd %ymm0,%ymm3,%ymm10,%ymm10
vaddpd %ymm10,%ymm4,%ymm4
cmp    $0x13880,%rax
jne    2420 <SACf__MAIN__acc_v___struct_Body_10000___struct_Body_10000___struct_Body_10000__d_10000+0x130>
