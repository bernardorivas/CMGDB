#ifndef CHOMP_RECTANGULAR_FIBER_H
#define CHOMP_RECTANGULAR_FIBER_H

#include <algorithm>
#include <optional>
#include <vector>
#include "chomp/CubicalComplex.h"

namespace chomp {

// Coordinates describe closed boxes on the vertex lattice. All covered
// top cubes must be present in the codomain and the cover is duplicate-free.
struct FiberBox {
  std::vector<uint64_t> low;
  std::vector<uint64_t> high;

  bool contains(const std::vector<uint64_t>& point, uint64_t axes) const {
    for (size_t i = 0; i < low.size(); ++i) {
      if (point[i] < low[i] || point[i] + ((axes >> i) & 1) > high[i]) return false;
    }
    return true;
  }
};

inline std::optional<FiberBox> CompleteFiberBox(
    const CubicalComplex& target, const std::vector<Index>& cubes) {
  if (cubes.empty()) return std::nullopt;
  auto corner = target.indexToCube(cubes.front());
  FiberBox box{corner, corner};
  for (Index cube : cubes) {
    auto point = target.indexToCube(cube);
    for (size_t i = 0; i < box.low.size(); ++i) {
      box.low[i] = std::min(box.low[i], point[i]);
      box.high[i] = std::max(box.high[i], point[i]);
    }
  }
  size_t volume = 1;
  for (size_t i = 0; i < box.low.size(); ++i) {
    ++box.high[i]; // Convert the last top cube to its upper vertex.
    const auto width = box.high[i] - box.low[i];
    if (volume > cubes.size() / width) return std::nullopt;
    volume *= width;
  }
  return volume == cubes.size() ? std::optional<FiberBox>(std::move(box)) : std::nullopt;
}

struct RectangularFiber {
  std::vector<const FiberBox*> x;
  std::vector<const FiberBox*> a;
  std::vector<uint64_t> anchor;

  bool relative() const { return !a.empty(); }

  bool contains(const std::vector<uint64_t>& point, uint64_t axes) const {
    auto touches = [&](const FiberBox* box) { return box->contains(point, axes); };
    return std::any_of(x.begin(), x.end(), touches) &&
           !std::any_of(a.begin(), a.end(), touches);
  }
};

template<class Neighbors, class Images, class Boxes>
std::optional<RectangularFiber> CommonFiberVertex(
    const Neighbors& X, const Neighbors& A, const Images& images, const Boxes& boxes) {
  RectangularFiber fiber;
  auto append = [&](const auto& neighbors, auto& output) {
    for (Index cell : neighbors) {
      if (images.at(cell).empty()) continue;
      const auto& box = boxes.at(cell);
      if (!box.has_value()) return false;
      output.push_back(&*box);
    }
    return true;
  };
  if (!append(X, fiber.x) || fiber.x.empty() || !append(A, fiber.a)) return std::nullopt;
  fiber.anchor = fiber.x.front()->low;
  auto upper = fiber.x.front()->high;
  // Exit neighbors are also in X. Thus this same vertex contracts both
  // unions, and the contraction descends to the relative chain complex.
  for (const FiberBox* box : fiber.x) {
    for (size_t d = 0; d < upper.size(); ++d) {
      fiber.anchor[d] = std::max(fiber.anchor[d], box->low[d]);
      upper[d] = std::min(upper[d], box->high[d]);
      if (fiber.anchor[d] > upper[d]) return std::nullopt;
    }
  }
  return fiber;
}

inline std::optional<Chain> RectangularPreboundary(
    const RectangularFiber& fiber, CubicalComplex& target, const Chain& input) {
  const int D = target.dimension();
  const int degree = input.dimension();
  const uint64_t mask = (uint64_t{1} << D) - 1;
  auto project = [&](const Chain& chain) {
    Chain result;
    result.dimension() = chain.dimension();
    for (const Term& term : chain()) {
      auto address = target.indexToCell(term.index(), result.dimension());
      if (fiber.contains(target.addressToCube(address), address & mask)) result += term;
    }
    return simplify(result);
  };
  Chain projected = project(input);
  Chain output;
  output.dimension() = degree + 1;
  for (const Term& term : projected()) {
    const auto address = target.indexToCell(term.index(), degree);
    auto point = target.addressToCube(address);
    const auto axes = address & mask;
    for (int d = 0; d < D; ++d) {
      const auto axis = uint64_t{1} << d;
      // Earlier coordinates have been collapsed. An edge in the current
      // coordinate vanishes under its projection, terminating the sum.
      if (axes & axis) break;
      const auto anchor = fiber.anchor[d];
      const auto original = point[d];
      const auto coefficient = original <= anchor ? term.coef() : -term.coef();
      for (auto q = std::min(original, anchor); q < std::max(original, anchor); ++q) {
        point[d] = q;
        if (!fiber.contains(point, axes | axis)) continue;
        const auto cell = target.cubeToAddress(point) | axes | axis;
        const auto index = target.cellToIndex(cell, degree + 1);
        if (index >= target.size(degree + 1)) return std::nullopt;
        output += Term(index, coefficient);
      }
      point[d] = anchor;
    }
  }
  output = simplify(output);
  if (!simplify(projected - project(target.boundary(output)))().empty()) return std::nullopt;
  return output;
}

} // namespace chomp
#endif
