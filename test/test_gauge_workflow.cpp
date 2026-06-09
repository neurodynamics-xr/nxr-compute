#include "nxr/facets.h"
#include <cmath>
#include <iostream>
using namespace nxr::manifold;

static int g_failures = 0;
#define EXPECT(cond, msg) do { \
    if (cond) { std::cout << "  [PASS] " << msg << "\n"; } \
    else { std::cout << "  [FAIL] " << msg << "\n"; ++g_failures; } } while (0)

static void icosphere(std::vector<double>& V, std::vector<int32_t>& F) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    V = {-1,t,0, 1,t,0, -1,-t,0, 1,-t,0, 0,-1,t, 0,1,t,
          0,-1,-t, 0,1,-t, t,0,-1, t,0,1, -t,0,-1, -t,0,1};
    F = {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2,
         10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5,
         2,4,11, 6,2,10, 8,6,7, 9,8,1};
}

static void testDefaultsAndValidation() {
    std::cout << "\n=== gauge: defaults + Gauss-Bonnet ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    EXPECT(m.activeGaugeType() == GaugeType::LeviCivita, "pattern 1 default = Levi-Civita");
    EXPECT(m.eulerCharacteristic() == 2, "chi = 2");

    // valid singularities (sum == chi == 2)
    bool ok = true;
    try { m.setGauge(GaugeType::Trivial, {{0,1.0},{3,1.0}}); }
    catch (...) { ok = false; }
    EXPECT(ok && m.activeGaugeType() == GaugeType::Trivial, "setGauge(trivial, sum=2) ok");

    // invalid sum throws
    bool threw = false;
    try { m.setGauge(GaugeType::Trivial, {{0,1.0}}); } catch (const Error&) { threw = true; }
    EXPECT(threw, "Gauss-Bonnet violation (sum=1 != chi=2) throws");

    // empty singularities throws
    threw = false;
    try { m.setGauge(GaugeType::Trivial, {}); } catch (const Error&) { threw = true; }
    EXPECT(threw, "trivial with empty singularities throws");

    // non-trivial gauge with singularities throws (fail loud, not silent discard)
    threw = false;
    try { m.setGauge(GaugeType::LeviCivita, {{0,1.0}}); } catch (const Error&) { threw = true; }
    EXPECT(threw, "non-trivial setGauge with singularities throws");
    // reset to a clean LC state for any later assertions
    m.setGauge(GaugeType::LeviCivita);
    EXPECT(m.activeGaugeType() == GaugeType::LeviCivita && m.activeSingularities().empty(),
           "setGauge(LeviCivita) clears to empty");

    // pattern 2: singularity ctor -> default trivial
    Manifold m2(V.data(), 12, F.data(), 20, std::map<int,double>{{0,1.0},{3,1.0}});
    EXPECT(m2.activeGaugeType() == GaugeType::Trivial, "pattern 2 ctor default = Trivial");
    // pattern 2 with bad sum throws at construction
    threw = false;
    try { Manifold mb(V.data(), 12, F.data(), 20, std::map<int,double>{{0,1.0}}); }
    catch (const Error&) { threw = true; }
    EXPECT(threw, "pattern 2 ctor validates Gauss-Bonnet");
}

int main() {
    testDefaultsAndValidation();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
