#pragma once
// Shared SDF math + world definition.
//
// Everything here is __host__ __device__, so the exact same distance field
// drives GPU rendering AND CPU-side physics / ray-gun picking. Only include
// this from .cu files (it needs nvcc).
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include "world_view.h"

#define HD __host__ __device__ inline

// =============================================================
// float3 helpers
// =============================================================
HD float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
HD float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
HD float3 operator-(float3 a) { return make_float3(-a.x, -a.y, -a.z); }
HD float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
HD float3 operator*(float s, float3 a) { return make_float3(a.x * s, a.y * s, a.z * s); }
HD float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
HD float3 operator/(float3 a, float s) { return make_float3(a.x / s, a.y / s, a.z / s); }
HD float3 operator/(float3 a, float3 b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
HD float3& operator+=(float3& a, float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
HD float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HD float length(float3 a) { return sqrtf(dot(a, a)); }
HD float3 normalize(float3 a) { float l = length(a); return l > 1e-8f ? a / l : make_float3(0.f, 0.f, 0.f); }
HD float3 abs3(float3 a) { return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z)); }
HD float clampf(float x, float a, float b) { return fmaxf(a, fminf(b, x)); }
HD float3 clamp3(float3 v, float a, float b) { return make_float3(clampf(v.x, a, b), clampf(v.y, a, b), clampf(v.z, a, b)); }
HD float mixf(float a, float b, float t) { return a + (b - a) * t; }
HD float3 mix3(float3 a, float3 b, float t) { return a + (b - a) * t; }
HD float fractf(float x) { return x - floorf(x); }
HD float smoothstepf(float e0, float e1, float x) {
    float t = clampf((x - e0) / (e1 - e0), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}

// =============================================================
// Hashing + 2D Perlin gradient noise
// =============================================================
HD uint32_t hash2i(int x, int y) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}
HD float hash2f(int x, int y) { return (float)(hash2i(x, y) & 0xffffffu) * (1.0f / 16777216.0f); }

HD float grad2(uint32_t h, float x, float y) {
    switch (h & 7u) {
        case 0: return  x + y;
        case 1: return  x - y;
        case 2: return -x + y;
        case 3: return -x - y;
        case 4: return  x;
        case 5: return -x;
        case 6: return  y;
        default: return -y;
    }
}

HD float perlin2(float x, float y) {
    float fx0 = floorf(x), fy0 = floorf(y);
    int ix = (int)fx0, iy = (int)fy0;
    float fx = x - fx0, fy = y - fy0;
    float u = fx * fx * fx * (fx * (fx * 6.f - 15.f) + 10.f);   // quintic fade
    float v = fy * fy * fy * (fy * (fy * 6.f - 15.f) + 10.f);
    float n00 = grad2(hash2i(ix,     iy),     fx,       fy);
    float n10 = grad2(hash2i(ix + 1, iy),     fx - 1.f, fy);
    float n01 = grad2(hash2i(ix,     iy + 1), fx,       fy - 1.f);
    float n11 = grad2(hash2i(ix + 1, iy + 1), fx - 1.f, fy - 1.f);
    return mixf(mixf(n00, n10, u), mixf(n01, n11, u), v);
}

// =============================================================
// Materials
// =============================================================
enum Material : int {
    MAT_NONE = 0,
    MAT_GRASS, MAT_DIRT, MAT_ROCK,
    MAT_BARK, MAT_LEAVES,
    MAT_LOG, MAT_ROOF, MAT_PLANK, MAT_STONE,
    MAT_TERRAIN = 100   // placeholder: resolved into grass/dirt/rock by terrainLayer()
};

// Soil layering (meters). Slopes are tan(angle).
constexpr float GRASS_THICKNESS = 0.035f;   // ~1.4 inch of lush grass
constexpr float DIRT_MIN        = 0.90f;    // ~3.0 feet of rich dirt (several feet deep)
constexpr float DIRT_MAX        = 1.50f;    // ~5.0 feet of rich dirt (several feet deep)
constexpr float GRASS_MAX_SLOPE = 0.58f;    // tan(30 deg): steeper -> no grass
constexpr float DIRT_MAX_SLOPE  = 0.84f;    // tan(40 deg): steeper -> no dirt, bare rock

// =============================================================
// SDF primitives
// =============================================================
HD float sdSphere(float3 p, float r) { return length(p) - r; }

HD float sdBox(float3 p, float3 b) {
    float3 q = abs3(p) - b;
    float3 qp = make_float3(fmaxf(q.x, 0.f), fmaxf(q.y, 0.f), fmaxf(q.z, 0.f));
    return length(qp) + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.f);
}

HD float sdRoundBox(float3 p, float3 b, float r) {
    return sdBox(p, make_float3(b.x - r, b.y - r, b.z - r)) - r;
}

// Capped cylinder along Y, centered at origin, half-height h.
HD float sdCylY(float3 p, float r, float h) {
    float dx = sqrtf(p.x * p.x + p.z * p.z) - r;
    float dy = fabsf(p.y) - h;
    float ox = fmaxf(dx, 0.f), oy = fmaxf(dy, 0.f);
    return fminf(fmaxf(dx, dy), 0.f) + sqrtf(ox * ox + oy * oy);
}

// Capped cylinder along Z, centered at origin, half-length h.
HD float sdCylZ(float3 p, float r, float h) {
    float dr = sqrtf(p.x * p.x + p.y * p.y) - r;
    float dz = fabsf(p.z) - h;
    float o1 = fmaxf(dr, 0.f), o2 = fmaxf(dz, 0.f);
    return fminf(fmaxf(dr, dz), 0.f) + sqrtf(o1 * o1 + o2 * o2);
}

// Torus whose axis is Z (ring lies in the XY plane).
HD float sdTorusZ(float3 p, float R, float r) {
    float qx = sqrtf(p.x * p.x + p.y * p.y) - R;
    return sqrtf(qx * qx + p.z * p.z) - r;
}

HD float sdCapsule(float3 p, float3 a, float3 b, float r) {
    float3 pa = p - a, ba = b - a;
    float h = clampf(dot(pa, ba) / dot(ba, ba), 0.f, 1.f);
    return length(pa - ba * h) - r;
}

// Capped cone along Y from y=-h (radius r1) to y=+h (radius r2). (Inigo Quilez)
HD float sdCappedCone(float3 p, float h, float r1, float r2) {
    float qx = sqrtf(p.x * p.x + p.z * p.z), qy = p.y;
    float k1x = r2, k1y = h;
    float k2x = r2 - r1, k2y = 2.f * h;
    float cax = qx - fminf(qx, (qy < 0.f) ? r1 : r2);
    float cay = fabsf(qy) - h;
    float t = clampf(((k1x - qx) * k2x + (k1y - qy) * k2y) / (k2x * k2x + k2y * k2y), 0.f, 1.f);
    float cbx = qx - k1x + k2x * t;
    float cby = qy - k1y + k2y * t;
    float s = (cbx < 0.f && cay < 0.f) ? -1.f : 1.f;
    return s * sqrtf(fminf(cax * cax + cay * cay, cbx * cbx + cby * cby));
}

// Smoothly subtract d1 from d2.
HD float opSmoothSubtract(float d1, float d2, float k) {
    float h = clampf(0.5f - 0.5f * (d2 + d1) / k, 0.f, 1.f);
    return (d2 * (1.f - h) - d1 * h) + k * h * (1.f - h);
}

// =============================================================
// Terrain: fBm Perlin heightfield, flattened into a pad under the cabin
// =============================================================
constexpr int   TERRAIN_OCTAVES   = 6;
constexpr float TERRAIN_LIPSCHITZ = 0.45f;   // heightfield is not a true SDF; scale steps down

HD float terrainRaw(float x, float z) {
    float h = 0.f, amp = 1.f, freq = 1.f / 70.f;
    float px = x, pz = z;
    for (int i = 0; i < TERRAIN_OCTAVES; ++i) {
        h += amp * perlin2(px * freq + 17.3f * i, pz * freq - 9.1f * i);
        // Rotate the domain each octave to hide grid-aligned artifacts.
        float nx = 0.8f * px - 0.6f * pz;
        float nz = 0.6f * px + 0.8f * pz;
        px = nx; pz = nz;
        amp *= 0.47f;
        freq *= 2.0f;
    }
    return h * 12.0f;
}

HD float terrainHeight(float x, float z, const WorldView& w) {
    float h = terrainRaw(x, z);
    float dx = x - w.cabin_x, dz = z - w.cabin_z;
    float m = smoothstepf(16.f, 7.f, sqrtf(dx * dx + dz * dz));
    return mixf(h, w.cabin_y, m);
}

HD float terrainSDF(float3 p, const WorldView& w) {
    return (p.y - terrainHeight(p.x, p.z, w)) * TERRAIN_LIPSCHITZ;
}

// Which soil layer is at point p? Depth is measured below the ORIGINAL surface,
// so carved crater walls expose the grass -> dirt -> rock strata.
HD int terrainLayer(float3 p, const WorldView& w) {
    const float e = 0.1f;
    float h  = terrainHeight(p.x, p.z, w);
    float hx = terrainHeight(p.x + e, p.z, w);
    float hz = terrainHeight(p.x, p.z + e, w);
    float slope = sqrtf((hx - h) * (hx - h) + (hz - h) * (hz - h)) / e;
    float depth = h - p.y;

    float n = perlin2(p.x * 0.7f, p.z * 0.7f);            // organic boundary wobble
    float s = slope + 0.08f * n;
    float grass = GRASS_THICKNESS * smoothstepf(GRASS_MAX_SLOPE, GRASS_MAX_SLOPE - 0.12f, s);
    float dirt  = mixf(DIRT_MIN, DIRT_MAX, 0.5f + 0.5f * n) * smoothstepf(DIRT_MAX_SLOPE, DIRT_MAX_SLOPE - 0.12f, s);

    if (depth < grass) return MAT_GRASS;
    if (depth < grass + dirt) return MAT_DIRT;
    return MAT_ROCK;
}

// =============================================================
// Trees: trunk + three stacked cones
// =============================================================
HD float treeSDF(float3 p, const TreeDesc& t, int* mat) {
    float3 q = make_float3(p.x - t.x, p.y - t.y, p.z - t.z);

    // Cheap bounding cylinder: a valid lower bound when we're far away.
    float bound = sdCylY(make_float3(q.x, q.y - 0.5f * t.height, q.z), t.radius + 0.25f, 0.5f * t.height + 0.6f);
    if (bound > 0.4f) return bound;

    float trunkR = 0.10f + 0.018f * t.height;
    float trunkTop = 0.55f * t.height;
    float trunk = sdCylY(make_float3(q.x, q.y - 0.5f * (trunkTop - 0.6f), q.z), trunkR, 0.5f * (trunkTop + 0.6f));

    float leaves = 1e9f;
    for (int k = 0; k < 3; ++k) {
        float fk = (float)k;
        float hc = t.height * (0.42f - 0.07f * fk);
        float y0 = t.height * (0.28f + 0.22f * fk);
        float r1 = t.radius * (1.0f - 0.24f * fk);
        float cone = sdCappedCone(make_float3(q.x, q.y - (y0 + 0.5f * hc), q.z), 0.5f * hc, r1, 0.06f * t.radius);
        leaves = fminf(leaves, cone);
    }
    if (mat) *mat = (trunk < leaves) ? MAT_BARK : MAT_LEAVES;
    return fminf(trunk, leaves);
}

HD float treesSDF(float3 p, const WorldView& w, int* mat) {
    float best = 1e9f;
    for (int i = 0; i < w.num_trees; ++i) {
        int m = MAT_LEAVES;
        float d = treeSDF(p, w.trees[i], mat ? &m : nullptr);
        if (d < best) { best = d; if (mat) *mat = m; }
    }
    return best;
}

// =============================================================
// Log cabin: interlocking log walls, gable roof, door, windows, chimney
// =============================================================
HD float cabinSDF(float3 p, const WorldView& w, int* mat) {
    float3 q = make_float3(p.x - w.cabin_x, p.y - (w.cabin_y + 0.25f), p.z - w.cabin_z);

    float bound = sdBox(q - make_float3(0.f, 2.2f, 0.f), make_float3(3.9f, 3.0f, 3.4f));
    if (bound > 0.5f) return bound;

    const float S = 0.30f;          // course spacing
    const float R = 0.165f;         // log radius (slight overlap between courses)
    const int   N = 9;              // courses
    const float WH = N * S;         // wall height

    // Logs running along X (front/back walls at z = +-2.5)
    float qzw = fabsf(q.z) - 2.5f;
    float i = clampf(floorf(q.y / S), 0.f, (float)(N - 1));
    float yy = q.y - (i + 0.5f) * S;
    float logsX = fmaxf(sqrtf(yy * yy + qzw * qzw) - R, fabsf(q.x) - 3.35f);

    // Logs running along Z (side walls at x = +-3.0), offset half a course so the corners interlock
    float qxw = fabsf(q.x) - 3.0f;
    float j = clampf(floorf(q.y / S - 0.5f), 0.f, (float)(N - 2));
    float yy2 = q.y - (j + 1.f) * S;
    float logsZ = fmaxf(sqrtf(yy2 * yy2 + qxw * qxw) - R, fabsf(q.z) - 2.85f);

    float logs = fminf(logsX, logsZ);

    // Door + window openings
    float door = sdBox(q - make_float3( 1.2f, 1.0f,  2.5f), make_float3(0.5f, 1.0f, 0.4f));
    float win1 = sdBox(q - make_float3(-1.4f, 1.4f,  2.5f), make_float3(0.5f, 0.4f, 0.4f));
    float win2 = sdBox(q - make_float3(-3.0f, 1.4f,  0.0f), make_float3(0.4f, 0.4f, 0.55f));
    float win3 = sdBox(q - make_float3( 0.0f, 1.4f, -2.5f), make_float3(0.6f, 0.4f, 0.4f));
    logs = fmaxf(logs, -fminf(fminf(door, win1), fminf(win2, win3)));

    // Gable roof: two slabs mirrored across z = 0
    const float CS = 0.857493f, SN = 0.514496f;     // cos/sin of atan(1.5 / 2.5)
    const float ridgeY = WH + 1.5f + 0.05f;
    float ry = q.y - ridgeY, rz = fabsf(q.z);
    float along = rz * CS - ry * SN;                // distance down the slope from the ridge
    float nrm   = rz * SN + ry * CS;                // height above the roof plane
    const float L = 3.15f / CS;                     // slab length incl. overhang
    float roof = sdBox(make_float3(q.x, along - 0.5f * L, nrm - 0.06f), make_float3(3.65f, 0.5f * L, 0.06f));

    // Triangular gable ends under the roof
    float gable = fmaxf(sdBox(make_float3(fabsf(q.x) - 3.0f, q.y - (WH + 0.85f), q.z), make_float3(0.1f, 0.85f, 2.6f)), nrm);

    float chimney  = sdBox(q - make_float3(-2.0f, 2.4f, -1.0f), make_float3(0.35f, 2.65f, 0.35f));
    float found    = sdBox(q - make_float3( 0.0f, -0.35f, 0.0f), make_float3(3.25f, 0.4f, 2.75f));
    float doorSlab = sdBox(q - make_float3( 1.2f, 1.0f, 2.42f), make_float3(0.47f, 0.98f, 0.04f));

    float d = logs; int m = MAT_LOG;
    if (roof < d)     { d = roof;     m = MAT_ROOF;  }
    if (gable < d)    { d = gable;    m = MAT_PLANK; }
    if (doorSlab < d) { d = doorSlab; m = MAT_PLANK; }
    if (chimney < d)  { d = chimney;  m = MAT_STONE; }
    if (found < d)    { d = found;    m = MAT_STONE; }
    if (mat) *mat = m;
    return d;
}

// =============================================================
// Ray-gun carves: smooth-subtract every carve sphere in this grid cell
// =============================================================
HD float applyCarves(float3 p, float d, const WorldView& w) {
    int cx = (int)floorf((p.x - EDIT_GRID_MIN) / EDIT_CELL);
    int cz = (int)floorf((p.z - EDIT_GRID_MIN) / EDIT_CELL);
    if (cx < 0 || cz < 0 || cx >= EDIT_GRID_N || cz >= EDIT_GRID_N) return d;
    int cell = cz * EDIT_GRID_N + cx;
    int n = w.cell_counts[cell];
    const int* items = w.cell_items + cell * EDIT_CELL_CAP;
    for (int i = 0; i < n; ++i) {
        float4 e = w.edits[items[i]];
        float ds = length(p - make_float3(e.x, e.y, e.z)) - e.w;
        d = opSmoothSubtract(ds, d, CARVE_SMOOTH);
    }
    return d;
}

// =============================================================
// The whole world
// =============================================================
HD float mapWorld(float3 p, const WorldView& w, int* mat = nullptr) {
    float d = terrainSDF(p, w);
    int m = MAT_TERRAIN;
    int mt = MAT_NONE;

    float dt = treesSDF(p, w, mat ? &mt : nullptr);
    if (dt < d) { d = dt; m = mt; }

    float dc = cabinSDF(p, w, mat ? &mt : nullptr);
    if (dc < d) { d = dc; m = mt; }

    // Carves apply to everything (terrain, trees, cabin). The material stays that of
    // the solid the carved surface cuts through, which is what exposes soil strata.
    d = applyCarves(p, d, w);
    if (mat) *mat = m;
    return d;
}
