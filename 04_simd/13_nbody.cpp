#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

float simd_sum(__m256 avec, int N) {
  float a[N];
  __m256 bvec = _mm256_permute2f128_ps(avec,avec,1);
  bvec = _mm256_add_ps(bvec,avec);
  bvec = _mm256_hadd_ps(bvec,bvec);
  bvec = _mm256_hadd_ps(bvec,bvec);
  _mm256_store_ps(a, bvec);
  return a[0];
}

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], cmp[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    cmp[i] = i;
  }
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  __m256 fxvec = _mm256_load_ps(fx);
  __m256 fyvec = _mm256_load_ps(fy);
  __m256 cmpvec = _mm256_load_ps(cmp);
  for(int i=0; i<N; i++) {
    __m256 ivec = _mm256_set1_ps(i);
    __m256 mask = _mm256_cmp_ps(ivec, cmpvec, _CMP_NEQ_OQ);
    __m256 xivec = _mm256_set1_ps(x[i]);
    __m256 yivec = _mm256_set1_ps(y[i]);
    __m256 rxvec = _mm256_sub_ps(xivec, xvec);
    __m256 ryvec = _mm256_sub_ps(yivec, yvec);
    __m256 rsqvec = _mm256_fmadd_ps(rxvec, rxvec, _mm256_mul_ps(ryvec, ryvec));
    rsqvec = _mm256_blendv_ps(_mm256_set1_ps(1), rsqvec, mask);
    __m256 rinvvec = _mm256_rsqrt_ps(rsqvec);
    __m256 rinvcube = _mm256_mul_ps(rinvvec, _mm256_mul_ps(rinvvec, rinvvec));
    __m256 fxivec = _mm256_mul_ps(_mm256_mul_ps(rxvec, mvec), rinvcube);
    fxivec = _mm256_blendv_ps(_mm256_setzero_ps(), fxivec, mask);
    __m256 fyivec = _mm256_mul_ps(_mm256_mul_ps(ryvec, mvec), rinvcube);
    fyivec = _mm256_blendv_ps(_mm256_setzero_ps(), fyivec, mask);
    fx[i] -= simd_sum(fxivec, N);
    fy[i] -= simd_sum(fyivec, N);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
