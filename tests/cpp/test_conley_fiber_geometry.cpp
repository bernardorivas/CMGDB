#include <cassert>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "chomp/RectangularFiber.h"

using namespace chomp;
using RingCounts = std::map<Index, Ring>;

// Iterate a small sparse lattice, including the closed upper vertex on
// nonperiodic axes and one-cube periodic axes where two incidences coincide.
void checkIncidences(const std::vector<uint64_t>& sizes, const std::vector<bool>& periodic) {
  CubicalComplex grid;
  grid.initialize(sizes, periodic);
  for (uint64_t i = 0; i < sizes[0]; ++i)
    for (uint64_t j = 0; j < sizes[1]; ++j)
      for (uint64_t k = 0; k < sizes[2]; ++k)
        if (i == 0 || j == 0 || k == 0) grid.addFullCube({i, j, k});
  grid.finalize();
  constexpr int D = 3;
  constexpr uint64_t axes_mask = 7;
  std::vector<std::vector<RingCounts>> expected_coboundary(D);
  for (int d = 0; d < D; ++d) expected_coboundary[d].resize(grid.size(d));
  for (int d = 0; d <= D; ++d) {
    for (Index cell = 0; cell < grid.size(d); ++cell) {
      const auto address = grid.indexToCell(cell, d);
      const auto point = grid.addressToCube(address);
      const auto axes = address & axes_mask;
      std::vector<Index> expected_cubes;
      for (Index cube = 0; cube < grid.size(D); ++cube) {
        const auto corner = grid.indexToCube(cube);
        bool incident = true;
        for (int axis = 0; axis < D; ++axis) {
          if (corner[axis] == point[axis]) continue;
          if (axes & (uint64_t{1} << axis)) {
            incident = false;
            break;
          }
          auto upper = corner[axis] + 1;
          if (periodic[axis]) upper %= sizes[axis];
          if (upper != point[axis]) {
            incident = false;
            break;
          }
        }
        if (incident) expected_cubes.push_back(cube);
      }
      assert(grid.incidentTopCubes(cell, d) == expected_cubes);

      Chain boundary = grid.boundary(cell, d);
      assert(boundary.dimension() == d - 1);
      std::vector<Term> expected_boundary;
      Ring sign(1);
      for (int axis = 0; axis < D; ++axis) {
        const auto bit = uint64_t{1} << axis;
        if (!(axes & bit)) continue;
        auto upper = point;
        ++upper[axis];
        if (periodic[axis]) upper[axis] %= sizes[axis];
        for (const auto& face : {point, upper}) {
          const auto facet = grid.cubeToAddress(face) | (axes ^ bit);
          const auto index = grid.cellToIndex(facet, d - 1);
          assert(index < grid.size(d - 1));
          expected_boundary.emplace_back(index, sign);
          auto& counts = expected_coboundary[d - 1][index];
          if (counts.count(cell) == 0) counts.emplace(cell, Ring(0));
          counts.at(cell) += sign;
          if (counts.at(cell) == Ring(0)) counts.erase(cell);
          sign = -sign;
        }
        sign = -sign;
      }
      assert(boundary() == expected_boundary);
    }
  }
  for (int d = 0; d < D; ++d) {
    for (Index cell = 0; cell < grid.size(d); ++cell) {
      RingCounts actual;
      const auto cofaces = simplify(grid.coboundary(cell, d));
      for (const auto& term : cofaces())
        actual.emplace(term.index(), term.coef());
      assert(actual == expected_coboundary[d][cell]);
    }
  }
}

void checkRectangularFibers() {
  CubicalComplex grid;
  grid.initialize({3, 3, 3}, {false, false, false});
  for (uint64_t i = 0; i < 3; ++i)
    for (uint64_t j = 0; j < 3; ++j)
      for (uint64_t k = 0; k < 3; ++k)
        grid.addFullCube({i, j, k});
  grid.finalize();

  std::unordered_map<Index, std::vector<Index>> rows;
  for (Index i = 0; i < grid.size(3); ++i) {
    const auto c = grid.indexToCube(i);
    rows[0].push_back(i);
    if (c[0] >= 1 && c[1] <= 1 && c[2] >= 1) rows[1].push_back(i);
    if (c == std::vector<uint64_t>{0, 0, 0}) {
      rows[2].push_back(i);
      rows[3].push_back(i);
    }
    if (c == std::vector<uint64_t>{2, 2, 2}) {
      rows[2].push_back(i);
      rows[4].push_back(i);
    }
    if (c[0] == 0) rows[5].push_back(i);
    if (c[1] == 0) rows[6].push_back(i);
  }
  std::unordered_map<Index, std::optional<FiberBox>> boxes;
  for (const auto& row : rows) boxes.emplace(row.first, CompleteFiberBox(grid, row.second));
  using Set = std::unordered_set<Index>;
  assert(!boxes.at(2));
  assert(!CommonFiberVertex(Set{3, 4}, Set{}, rows, boxes));
  auto absolute = CommonFiberVertex(Set{0}, Set{}, rows, boxes);
  auto relative = CommonFiberVertex(Set{0, 1}, Set{1}, rows, boxes);
  auto union_of_boxes = CommonFiberVertex(Set{5, 6}, Set{5}, rows, boxes);
  assert(absolute && !absolute->relative());
  assert(relative && relative->relative());
  assert(union_of_boxes && union_of_boxes->relative());
  absolute->anchor = {1, 1, 1};
  for (const auto& fiber : {*absolute, *relative})
    for (int d = 1; d <= 3; ++d)
      for (Index i = 0; i < grid.size(d); ++i)
        assert(RectangularPreboundary(fiber, grid, grid.boundary(i, d)));
  for (int d = 1; d <= 3; ++d) {
    for (Index i = 0; i < grid.size(d); ++i) {
      const auto address = grid.indexToCell(i, d);
      if (union_of_boxes->contains(grid.addressToCube(address), address & 7))
        assert(RectangularPreboundary(*union_of_boxes, grid, grid.boundary(i, d)));
    }
  }

  Chain lone_vertex;
  lone_vertex.dimension() = 0;
  lone_vertex += Term(0, Ring(1));
  assert(!RectangularPreboundary(*absolute, grid, lone_vertex));
}

void checkAddressIndexUpdates() {
  CubicalComplex grid;
  grid.initialize({2}, {false});
  grid.addFullCube({0});
  grid.addFullCube({1});
  grid.finalize();
  const auto vertex = grid.cellToIndex(grid.cubeToAddress({1}), 0);
  const auto second_cube_address = grid.indexToCell(grid.cubeIndex({1}), 1);
  assert(grid.incidentTopCubes(vertex, 0).size() == 2);
  grid.erase(second_cube_address);
  assert(grid.incidentTopCubes(vertex, 0).size() == 1);
  grid.finalize();
  assert(grid.incidentTopCubes(vertex, 0).size() == 1);

  CubicalComplex sparse;
  sparse.initialize({1024, 1024, 1024}, {false, false, false});
  sparse.addFullCube({0, 0, 0});
  sparse.addFullCube({1023, 1023, 1023});
  sparse.finalize();
  const auto far_corner = sparse.cellToIndex(sparse.cubeToAddress({1024, 1024, 1024}), 0);
  const auto far_cube = sparse.cubeIndex({1023, 1023, 1023});
  assert(sparse.incidentTopCubes(far_corner, 0) == std::vector<Index>{far_cube});
}

int main() {
  checkIncidences({3, 2, 2}, {false, false, false});
  checkIncidences({3, 2, 1}, {true, false, true});
  checkIncidences({1, 2, 3}, {true, true, false});
  checkRectangularFibers();
  checkAddressIndexUpdates();
}
