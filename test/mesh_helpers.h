// Shared by the eigenmode tests: the unit icosphere at a subdivision level, and a tiny pass/fail macro.
#pragma once
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

static int g_failures = 0;
#define EXPECT(cond, msg) do { \
    if (cond) { std::cout << "  [PASS] " << msg << "\n"; } \
    else { std::cout << "  [FAIL] " << msg << "\n"; ++g_failures; } } while (0)

inline void icosphere(int level, std::vector<double>& V, std::vector<int32_t>& F, double radius = 1.0) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    V = {-1,t,0, 1,t,0, -1,-t,0, 1,-t,0, 0,-1,t, 0,1,t, 0,-1,-t, 0,1,-t, t,0,-1, t,0,1, -t,0,-1, -t,0,1};
    F = {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5,
         2,4,11, 6,2,10, 8,6,7, 9,8,1};
    auto norm = [&](int i) { double l = std::sqrt(V[3*i]*V[3*i]+V[3*i+1]*V[3*i+1]+V[3*i+2]*V[3*i+2]); for (int k = 0; k < 3; ++k) V[3*i+k] /= l; };
    for (int i = 0; i < 12; ++i) norm(i);
    for (int s = 0; s < level; ++s) {
        std::map<std::pair<int,int>, int> mid;
        auto midpoint = [&](int a, int b) {
            auto key = std::make_pair(std::min(a, b), std::max(a, b));
            auto it = mid.find(key);
            if (it != mid.end()) return it->second;
            int n = static_cast<int>(V.size() / 3);
            for (int k = 0; k < 3; ++k) V.push_back(0.5 * (V[3*a+k] + V[3*b+k]));
            norm(n);
            return mid[key] = n;
        };
        std::vector<int32_t> G;
        for (size_t f = 0; f < F.size(); f += 3) {
            int a = F[f], b = F[f+1], c = F[f+2], ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
            for (int x : {a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca}) G.push_back(x);
        }
        F = G;
    }
    for (double& x : V) x *= radius;
}
