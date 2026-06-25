#pragma once
#include "nxr/operator_registry.h"   // reuse Domain, Bundle, FieldType, Gauge, OperatorVariant API
#include <string>
#include <string_view>
#include <vector>
#include <functional>

namespace nxr::manifold::registry {

// New axes the operator registry didn't need (operators don't carry n-form degree
// or a component-layout representation; fields do).
enum class NForm          { na, zero, one, two };
enum class Representation  { na, world, local_frame, intrinsic_complex, quaternion_interleaved };

inline const char* toString(NForm v)         { static const char* s[]{"na","zero","one","two"}; return s[(int)v]; }
inline const char* toString(Representation v) { static const char* s[]{"na","world","local_frame","intrinsic_complex","quaternion_interleaved"}; return s[(int)v]; }

// The spatial type of a field. No time axis by design: a Field is "a typed payload
// over a domain", never "a vector" — a future temporal subsystem composes OVER this.
struct FieldDescriptor {
    Domain         domain;
    Bundle         bundle;
    FieldType      field_type;
    NForm          n_form         = NForm::na;
    Representation  representation = Representation::na;
    Gauge          gauge          = Gauge::na;   // which coordinate system local_frame/intrinsic_complex references
    int            nSym           = 1;           // n-RoSy order (tangent); advisory parameter, not a match key
};

struct FieldVariant {
    std::string     id;
    std::string     label;
    FieldDescriptor descriptor;
    std::string     notes;
};

// A declared representation conversion, wired to an existing implementation.
struct ConversionEdge {
    std::string from;          // Field-variant id
    std::string to;            // Field-variant id
    std::string impl;          // existing function name ("" if declared-only)
    bool        implemented;   // false => declared, unimplemented
};

// ── Catalogue ──
const std::vector<FieldVariant>& fieldRegistry();
const FieldVariant*  fieldById(std::string_view id);
std::vector<const FieldVariant*>
                     fieldsWhere(const std::function<bool(const FieldVariant&)>& pred);

// ── Routing (implemented in a later task) ──
// Structural match: domain + bundle + field_type + n_form + representation.
// gauge and nSym are advisory parameters, NOT match keys — so e.g. a trivial-gauge
// tangent field still matches a Levi-Civita connection operator's input. requireField
// is a bundle/domain/degree/representation gate, not a gauge gate.
bool fieldMatches(const FieldDescriptor& f, const FieldDescriptor& expected);
void requireField(const FieldDescriptor& f, std::string_view operatorId);
std::vector<std::string> operatorsAccepting(const FieldDescriptor& f);
int componentsPerElement(const FieldDescriptor& f);
// totalScalars is the FLAT scalar count of the field, i.e. nElements(domain) *
// componentsPerElement — VectorXd.size(), or rows*cols of a flattened multi-component
// field. NOT a 2D MatrixXd's .rows() alone (an [nV×3] ambient field has 3*nV scalars).
void validateFieldShape(const FieldDescriptor& f, int totalScalars, int nV, int nE, int nF);

// ── Conversion graph (implemented in a later task) ──
const std::vector<ConversionEdge>& conversionGraph();

} // namespace nxr::manifold::registry
