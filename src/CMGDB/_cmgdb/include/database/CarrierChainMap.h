// CarrierChainMap.h
//
// Chain maps induced by acyclic carriers between simplicial complexes over
// GF(5): the kernel of CMGDB.ComputeCarrierChainMap.
//
// A simplicial complex is given by its cells in complex order: sorted by
// dimension, then lexicographically by the sorted tuple of vertex labels.
// Every source vertex has a set of target vertices, its vertex image. The
// carrier of a source cell s is the target subcomplex induced on T(s), the
// union of the vertex images of the vertices of s.
//
// The kernel checks, in this order, that every carrier is nonempty, that
// every distinct carrier is acyclic over GF(5), and that the carrier of every
// cell of the source P0 (the subcomplex induced on the exit vertices) lies in
// the target P0. Each check reports the first failing source cell in complex
// order. Acyclicity is checked once for every distinct carrier, in order of
// first use, and the kernel stops at the first cell whose carrier is not
// acyclic. It then constructs the canonical chain selector:
//   - a vertex v goes to the smallest vertex of T(v);
//   - a d-cell s (d >= 1) goes to the unique d-chain c of its carrier with
//     boundary ( c ) = phi ( boundary ( s ) ) that is supported on the greedy
//     independent d-cells of the carrier: the d-cells, in complex order, whose
//     boundary is not in the span of the boundaries of the earlier ones.
// The boundary columns of a carrier are reduced as in hybrid_dynamics
// (smallest pivot row first, with the combination of original columns kept at
// every pivot), and the entries of every image are listed in the order in
// which that reduction produces them. The sparse output therefore reproduces
// the Python reference entry for entry, including the order of the entries.
// Finally the chain map is validated: d phi = phi d over GF(5), every image
// lies in its carrier, and phi maps P0 into P0.
//
// The field arithmetic is local to this header; CHomP's Smith normal form is
// not used.

#ifndef CMDB_CARRIER_CHAIN_MAP_H
#define CMDB_CARRIER_CHAIN_MAP_H

#include <stdint.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "GF5.h"

namespace carrier_chain_map {

using gf5::Add5;
using gf5::Inverse5;
using gf5::Multiply5;
using gf5::Negate5;

/// Incidence (-1)^i of the face obtained by removing vertex position i.
inline uint8_t Incidence5 ( int64_t i ) {
  return ( i % 2 == 0 ) ? 1 : 4;
}

/// Largest ratio of the range of the vertex labels to the number of vertices
/// for which the rows of the labels are kept in a table. The table then takes
/// at most 128 bytes per vertex; sparser labels are found by binary search.
const int64_t kLabelTableRatio = 32;

/// A simplicial complex in complex order. Vertices are identified by their
/// rows among the 0-cells; since the 0-cells are sorted by label, the order of
/// vertex rows is the order of labels.
struct SimplicialComplex {
  std::vector<int32_t> labels;                // labels of the 0-cells, increasing
  std::vector<int64_t> counts;                // counts [ d ]: number of d-cells
  std::vector<std::vector<int32_t> > cells;   // counts [ d ] x ( d + 1 ) vertex rows
  std::vector<std::vector<int32_t> > faces;   // counts [ d ] x ( d + 1 ) rows of (d-1)-cells
  std::vector<std::vector<int64_t> > first;   // counts [ 0 ] + 1 offsets of d-cells by first vertex
  // label_rows [ label - label_low ]: row of the vertex with that label, or
  // -1; empty when the labels are searched instead.
  int64_t label_low = 0;
  std::vector<int32_t> label_rows;

  /// Row of the vertex with the given label, or -1.
  int32_t vertex_row ( int32_t label ) const {
    if ( ! label_rows . empty () ) {
      const int64_t offset = static_cast<int64_t> ( label ) - label_low;
      if ( offset < 0 || offset >= static_cast<int64_t> ( label_rows . size () ) ) {
        return -1;
      }
      return label_rows [ offset ];
    }
    std::vector<int32_t>::const_iterator found =
      std::lower_bound ( labels . begin (), labels . end (), label );
    if ( found == labels . end () || * found != label ) return -1;
    return static_cast<int32_t> ( found - labels . begin () );
  }

  int64_t degrees ( void ) const {
    return static_cast<int64_t> ( counts . size () );
  }

  int64_t count ( int64_t d ) const {
    return d < degrees () ? counts [ d ] : 0;
  }

  const int32_t * cell ( int64_t d, int64_t row ) const {
    return cells [ d ] . data () + row * ( d + 1 );
  }

  const int32_t * face_rows ( int64_t d, int64_t row ) const {
    return faces [ d ] . data () + row * ( d + 1 );
  }
};

inline std::string DescribeTuple ( const std::vector<int32_t> & labels,
                                   const int32_t * rows,
                                   int64_t width ) {
  std::ostringstream text;
  text << "(";
  for ( int64_t k = 0; k < width; ++ k ) {
    if ( k > 0 ) text << ", ";
    text << labels [ rows [ k ] ];
  }
  if ( width == 1 ) text << ",";
  text << ")";
  return text . str ();
}

/// Row of the d-cell with the given vertex rows, or -1. The first vertex
/// offsets of degree d must be built.
inline int64_t FindCell ( const SimplicialComplex & complex,
                          int64_t d,
                          const int32_t * vertices ) {
  const int64_t width = d + 1;
  int64_t low = complex . first [ d ] [ vertices [ 0 ] ];
  int64_t high = complex . first [ d ] [ vertices [ 0 ] + 1 ];
  while ( low < high ) {
    const int64_t middle = low + ( high - low ) / 2;
    const int32_t * candidate = complex . cell ( d, middle );
    if ( std::lexicographical_compare ( candidate, candidate + width,
                                        vertices, vertices + width ) ) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  if ( low < complex . first [ d ] [ vertices [ 0 ] + 1 ] &&
       std::equal ( vertices, vertices + width, complex . cell ( d, low ) ) ) {
    return low;
  }
  return -1;
}

/// Validate the cells of one complex (complex order, closure under faces)
/// and index them.
inline void BuildSimplicialComplex (
    const std::vector<std::vector<int32_t> > & simplices,
    const std::string & name,
    SimplicialComplex & complex ) {
  if ( simplices . empty () ) {
    throw std::invalid_argument (
      name + " must contain the array of 0-cells" );
  }
  const int64_t degrees = static_cast<int64_t> ( simplices . size () );
  complex . counts . assign ( degrees, 0 );
  complex . cells . assign ( degrees, std::vector<int32_t> () );
  complex . faces . assign ( degrees, std::vector<int32_t> () );
  complex . first . assign ( degrees, std::vector<int64_t> () );

  complex . labels = simplices [ 0 ];
  const std::vector<int32_t> & labels = complex . labels;
  const int64_t vertices = static_cast<int64_t> ( labels . size () );
  for ( int64_t d = 0; d < degrees; ++ d ) {
    const int64_t width = d + 1;
    if ( simplices [ d ] . size () % width != 0 ) {
      std::ostringstream message;
      message << name << "[" << d << "] must have " << width
              << " vertices in every row";
      throw std::invalid_argument ( message . str () );
    }
    const int64_t count = static_cast<int64_t> ( simplices [ d ] . size () ) / width;
    if ( count > static_cast<int64_t> ( std::numeric_limits<int32_t>::max () ) ) {
      std::ostringstream message;
      message << name << "[" << d << "] has more than 2^31 - 1 cells";
      throw std::overflow_error ( message . str () );
    }
    complex . counts [ d ] = count;
  }

  for ( int64_t row = 1; row < vertices; ++ row ) {
    if ( labels [ row ] <= labels [ row - 1 ] ) {
      std::ostringstream message;
      message << name << "[0] rows " << row - 1 << " and " << row
              << " (vertex labels " << labels [ row - 1 ] << " and "
              << labels [ row ]
              << ") are not in strictly increasing order";
      throw std::invalid_argument ( message . str () );
    }
  }
  complex . cells [ 0 ] . resize ( vertices );
  for ( int64_t row = 0; row < vertices; ++ row ) {
    complex . cells [ 0 ] [ row ] = static_cast<int32_t> ( row );
  }
  complex . label_low = 0;
  complex . label_rows . clear ();
  if ( vertices > 0 ) {
    const int64_t low = labels . front ();
    const int64_t range = static_cast<int64_t> ( labels . back () ) - low + 1;
    if ( range <= kLabelTableRatio * vertices ) {
      complex . label_low = low;
      complex . label_rows . assign ( range, -1 );
      for ( int64_t row = 0; row < vertices; ++ row ) {
        complex . label_rows [ labels [ row ] - low ] = static_cast<int32_t> ( row );
      }
    }
  }

  // Vertex rows of every cell, and the complex order.
  for ( int64_t d = 1; d < degrees; ++ d ) {
    const int64_t width = d + 1;
    const int64_t count = complex . counts [ d ];
    const std::vector<int32_t> & input = simplices [ d ];
    std::vector<int32_t> & cells = complex . cells [ d ];
    cells . resize ( input . size () );
    for ( int64_t row = 0; row < count; ++ row ) {
      for ( int64_t k = 0; k < width; ++ k ) {
        const int32_t label = input [ row * width + k ];
        const int32_t vertex = complex . vertex_row ( label );
        if ( vertex < 0 ) {
          std::ostringstream message;
          message << name << "[" << d << "] row " << row
                  << " has vertex label " << label
                  << ", which is not a 0-cell of " << name;
          throw std::invalid_argument ( message . str () );
        }
        cells [ row * width + k ] = vertex;
        if ( k > 0 && cells [ row * width + k ] <= cells [ row * width + k - 1 ] ) {
          std::ostringstream message;
          message << name << "[" << d << "] row " << row
                  << " is not a strictly increasing vertex tuple";
          throw std::invalid_argument ( message . str () );
        }
      }
      if ( row > 0 &&
           ! std::lexicographical_compare (
               cells . begin () + ( row - 1 ) * width,
               cells . begin () + row * width,
               cells . begin () + row * width,
               cells . begin () + ( row + 1 ) * width ) ) {
        std::ostringstream message;
        message << name << "[" << d << "] rows " << row - 1 << " and " << row
                << " are not in strictly increasing lexicographic order";
        throw std::invalid_argument ( message . str () );
      }
    }
  }

  // Cells grouped by first vertex; the complex order makes each group a
  // contiguous range of rows.
  for ( int64_t d = 0; d < degrees; ++ d ) {
    const int64_t width = d + 1;
    std::vector<int64_t> & first = complex . first [ d ];
    first . assign ( vertices + 1, 0 );
    for ( int64_t row = 0; row < complex . counts [ d ]; ++ row ) {
      ++ first [ complex . cells [ d ] [ row * width ] + 1 ];
    }
    for ( int64_t v = 0; v < vertices; ++ v ) first [ v + 1 ] += first [ v ];
  }

  // Faces, by removal index.
  std::vector<int32_t> face ( degrees );
  for ( int64_t d = 1; d < degrees; ++ d ) {
    const int64_t width = d + 1;
    const int64_t count = complex . counts [ d ];
    std::vector<int32_t> & faces = complex . faces [ d ];
    faces . resize ( count * width );
    for ( int64_t row = 0; row < count; ++ row ) {
      const int32_t * cell = complex . cell ( d, row );
      for ( int64_t i = 0; i < width; ++ i ) {
        int64_t position = 0;
        for ( int64_t k = 0; k < width; ++ k ) {
          if ( k != i ) face [ position ++ ] = cell [ k ];
        }
        const int64_t found = FindCell ( complex, d - 1, face . data () );
        if ( found < 0 ) {
          std::ostringstream message;
          message << "face " << DescribeTuple ( labels, face . data (), d )
                  << " (removal index " << i << ") of " << name << "["
                  << d << "] row " << row << " is not a cell of " << name
                  << "[" << d - 1 << "]";
          throw std::invalid_argument ( message . str () );
        }
        faces [ row * width + i ] = static_cast<int32_t> ( found );
      }
    }
  }
}

/// A sparse vector over GF(5) indexed by 0..size-1 that lists its entries in
/// the insertion order of a Python dict: a new key is appended, a key whose
/// value becomes zero is removed, and a removed key that returns is appended
/// again.
class OrderedChain {
public:
  void reserve ( int64_t size ) {
    if ( static_cast<int64_t> ( value_ . size () ) < size ) {
      value_ . resize ( size, 0 );
      position_ . resize ( size, -1 );
    }
  }

  void add ( int32_t key, uint8_t delta ) {
    const uint8_t sum = Add5 ( value_ [ key ], delta );
    value_ [ key ] = sum;
    if ( sum == 0 ) {
      position_ [ key ] = -1;
    } else if ( position_ [ key ] < 0 ) {
      position_ [ key ] = static_cast<int64_t> ( order_ . size () );
      order_ . push_back ( key );
    }
  }

  /// Visit the nonzero entries in insertion order, then clear.
  template < class Visitor >
  void drain ( Visitor visit ) {
    for ( size_t i = 0; i < order_ . size (); ++ i ) {
      const int32_t key = order_ [ i ];
      if ( position_ [ key ] == static_cast<int64_t> ( i ) ) {
        visit ( key, value_ [ key ] );
      }
    }
    clear ();
  }

  void clear ( void ) {
    for ( size_t i = 0; i < order_ . size (); ++ i ) {
      value_ [ order_ [ i ] ] = 0;
      position_ [ order_ [ i ] ] = -1;
    }
    order_ . clear ();
  }

private:
  std::vector<uint8_t> value_;
  std::vector<int64_t> position_;
  std::vector<int32_t> order_;
};

/// Column echelon form of a boundary matrix over GF(5), in the scheme of
/// hybrid_dynamics' _eliminate_columns_mod_prime and _solve_with_pivots.
///
/// Columns are added in order. A column is reduced at its smallest nonzero
/// row by the stored vector with that pivot until the pivot is new; the
/// vector is then stored, scaled to one at its pivot, with the combination of
/// original columns it equals. The stored columns are the greedy independent
/// columns. A right-hand side is reduced the same way; the solution is the
/// accumulated combination, supported on the stored columns, and it lists its
/// entries in the order in which the Python reduction inserts them.
class ColumnEliminator {
public:
  void reset ( int64_t rows, int64_t columns, bool combinations ) {
    combinations_ = combinations;
    if ( static_cast<int64_t> ( work_ . size () ) < rows ) {
      work_ . resize ( rows, 0 );
    }
    pivot_of_ . assign ( rows, -1 );
    vector_begin_ . assign ( 1, 0 );
    vector_rows_ . clear ();
    vector_values_ . clear ();
    combination_begin_ . assign ( 1, 0 );
    combination_columns_ . clear ();
    combination_values_ . clear ();
    if ( combinations ) chain_ . reserve ( columns );
  }

  int64_t rank ( void ) const {
    return static_cast<int64_t> ( vector_begin_ . size () ) - 1;
  }

  /// Add column number `column`, the boundary of a d-cell whose faces, by
  /// removal index, are the local rows `rows [ 0 .. width )`. Returns whether
  /// the column is independent of the earlier ones.
  bool add_column ( int32_t column, const int32_t * rows, int64_t width ) {
    heap_ . clear ();
    for ( int64_t i = 0; i < width; ++ i ) {
      accumulate ( rows [ i ], Incidence5 ( i ) );
    }
    if ( combinations_ ) chain_ . add ( column, 1 );
    while ( ! heap_ . empty () ) {
      const int32_t pivot = pop_minimum ();
      const uint8_t value = work_ [ pivot ];
      if ( value == 0 ) continue;
      const int64_t stored = pivot_of_ [ pivot ];
      if ( stored >= 0 ) {
        const uint8_t scale = Negate5 ( value );
        add_stored_vector ( stored, scale );
        if ( combinations_ ) add_stored_combination ( stored, scale );
        continue;
      }
      // A new pivot: every nonzero row is at or after it, and in the heap.
      const uint8_t inverse = Inverse5 ( value );
      vector_rows_ . push_back ( pivot );
      vector_values_ . push_back ( 1 );
      work_ [ pivot ] = 0;
      while ( ! heap_ . empty () ) {
        const int32_t row = pop_minimum ();
        if ( work_ [ row ] == 0 ) continue;
        vector_rows_ . push_back ( row );
        vector_values_ . push_back ( Multiply5 ( work_ [ row ], inverse ) );
        work_ [ row ] = 0;
      }
      vector_begin_ . push_back ( static_cast<int64_t> ( vector_rows_ . size () ) );
      if ( combinations_ ) {
        chain_ . drain ( [ & ] ( int32_t key, uint8_t coefficient ) {
          combination_columns_ . push_back ( key );
          combination_values_ . push_back ( Multiply5 ( coefficient, inverse ) );
        } );
        combination_begin_ . push_back (
          static_cast<int64_t> ( combination_columns_ . size () ) );
      }
      pivot_of_ [ pivot ] = rank () - 1;
      return true;
    }
    if ( combinations_ ) chain_ . clear ();
    return false;
  }

  /// Solve boundary ( c ) = rhs for c supported on the stored columns. The
  /// right-hand side has distinct local rows and nonzero values. Returns false
  /// when there is no solution.
  bool solve ( const std::vector<std::pair<int32_t, uint8_t> > & rhs,
               std::vector<std::pair<int32_t, uint8_t> > & solution ) {
    if ( ! combinations_ ) {
      throw std::runtime_error (
        "internal error: solving requires the column combinations" );
    }
    solution . clear ();
    heap_ . clear ();
    for ( size_t e = 0; e < rhs . size (); ++ e ) {
      accumulate ( rhs [ e ] . first, rhs [ e ] . second );
    }
    while ( ! heap_ . empty () ) {
      const int32_t pivot = pop_minimum ();
      const uint8_t value = work_ [ pivot ];
      if ( value == 0 ) continue;
      const int64_t stored = pivot_of_ [ pivot ];
      if ( stored < 0 ) {
        work_ [ pivot ] = 0;
        while ( ! heap_ . empty () ) work_ [ pop_minimum () ] = 0;
        chain_ . clear ();
        return false;
      }
      add_stored_vector ( stored, Negate5 ( value ) );
      add_stored_combination ( stored, value );
    }
    chain_ . drain ( [ & ] ( int32_t key, uint8_t coefficient ) {
      solution . push_back ( std::make_pair ( key, coefficient ) );
    } );
    return true;
  }

private:
  void accumulate ( int32_t row, uint8_t delta ) {
    const uint8_t before = work_ [ row ];
    const uint8_t after = Add5 ( before, delta );
    work_ [ row ] = after;
    if ( before == 0 && after != 0 ) {
      heap_ . push_back ( row );
      std::push_heap ( heap_ . begin (), heap_ . end (), std::greater<int32_t> () );
    }
  }

  int32_t pop_minimum ( void ) {
    std::pop_heap ( heap_ . begin (), heap_ . end (), std::greater<int32_t> () );
    const int32_t row = heap_ . back ();
    heap_ . pop_back ();
    return row;
  }

  void add_stored_vector ( int64_t stored, uint8_t scale ) {
    for ( int64_t e = vector_begin_ [ stored ]; e < vector_begin_ [ stored + 1 ]; ++ e ) {
      accumulate ( vector_rows_ [ e ], Multiply5 ( scale, vector_values_ [ e ] ) );
    }
  }

  void add_stored_combination ( int64_t stored, uint8_t scale ) {
    for ( int64_t e = combination_begin_ [ stored ];
          e < combination_begin_ [ stored + 1 ]; ++ e ) {
      chain_ . add ( combination_columns_ [ e ],
                     Multiply5 ( scale, combination_values_ [ e ] ) );
    }
  }

  bool combinations_ = false;
  std::vector<uint8_t> work_;          // dense working vector, zero between uses
  std::vector<int32_t> heap_;          // min-heap of rows that became nonzero
  std::vector<int64_t> pivot_of_;      // local row -> stored vector, or -1
  std::vector<int64_t> vector_begin_;
  std::vector<int32_t> vector_rows_;
  std::vector<uint8_t> vector_values_;
  std::vector<int64_t> combination_begin_;
  std::vector<int32_t> combination_columns_;
  std::vector<uint8_t> combination_values_;
  OrderedChain chain_;
};

/// Distinct carriers, identified by their sorted target vertex rows and
/// numbered in order of first use.
class CarrierTable {
public:
  CarrierTable ( void ) : offsets_ ( 1, 0 ) {}

  /// Id of the carrier with these sorted vertex rows; `inserted` tells
  /// whether it is new.
  int64_t intern ( const std::vector<int32_t> & vertices, bool & inserted ) {
    uint64_t hash = 0x9E3779B97F4A7C15ULL ^ static_cast<uint64_t> ( vertices . size () );
    for ( size_t i = 0; i < vertices . size (); ++ i ) {
      hash ^= static_cast<uint64_t> ( static_cast<uint32_t> ( vertices [ i ] ) );
      hash *= 0x100000001B3ULL;
      hash ^= hash >> 29;
    }
    std::unordered_map<uint64_t, int64_t>::iterator bucket = head_ . find ( hash );
    if ( bucket != head_ . end () ) {
      for ( int64_t id = bucket -> second; id >= 0; id = next_ [ id ] ) {
        if ( size ( id ) == static_cast<int64_t> ( vertices . size () ) &&
             std::equal ( vertices . begin (), vertices . end (), begin ( id ) ) ) {
          inserted = false;
          return id;
        }
      }
    }
    const int64_t id = count ();
    vertices_ . insert ( vertices_ . end (), vertices . begin (), vertices . end () );
    offsets_ . push_back ( static_cast<int64_t> ( vertices_ . size () ) );
    if ( bucket != head_ . end () ) {
      next_ . push_back ( bucket -> second );
      bucket -> second = id;
    } else {
      next_ . push_back ( -1 );
      head_ [ hash ] = id;
    }
    inserted = true;
    return id;
  }

  int64_t count ( void ) const {
    return static_cast<int64_t> ( offsets_ . size () ) - 1;
  }

  int64_t size ( int64_t id ) const {
    return offsets_ [ id + 1 ] - offsets_ [ id ];
  }

  const int32_t * begin ( int64_t id ) const {
    return vertices_ . data () + offsets_ [ id ];
  }

  const int32_t * end ( int64_t id ) const {
    return vertices_ . data () + offsets_ [ id + 1 ];
  }

private:
  std::vector<int64_t> offsets_;
  std::vector<int32_t> vertices_;
  std::unordered_map<uint64_t, int64_t> head_;
  std::vector<int64_t> next_;
};

/// Default limits of the first block of source cells whose carriers are
/// interned together before the new ones are checked for acyclicity: its
/// number of cells, and the total number of vertices of their carriers. The
/// limits double from one block to the next.
const int64_t kCarrierBlockCells = 4096;
const int64_t kCarrierBlockVertices = int64_t ( 1 ) << 20;

struct CarrierChainMapInput {
  std::vector<std::vector<int32_t> > source_simplices;   // labels, flat per degree
  std::vector<std::vector<int32_t> > target_simplices;   // unused when target_is_source
  bool target_is_source = true;
  std::vector<int64_t> vertex_image_indptr;
  std::vector<int32_t> vertex_image_indices;
  std::vector<uint8_t> source_exit;
  std::vector<uint8_t> target_exit;                      // unused when target_is_source
  bool return_chain_map = true;                          // form the result's chain_map
  // Limits of the first block, raised to 1 if smaller; the result does not
  // depend on them.
  int64_t block_cells = kCarrierBlockCells;
  int64_t block_vertices = kCarrierBlockVertices;
};

typedef std::tuple<uint64_t, uint64_t, int> PayloadEntry;

struct CarrierChainMapResult {
  std::string status = "ok";
  int64_t failure_degree = -1;
  int64_t failure_row = -1;
  int64_t carrier_count = 0;
  bool has_chain_map = false;
  // chain_map [ d ]: flat ( source_row, target_row, coefficient ) triples.
  std::vector<std::vector<int64_t> > chain_map;
  bool has_payload = false;
  std::vector<uint64_t> cell_counts;
  std::vector<std::vector<PayloadEntry> > boundary_entries;
  std::vector<std::vector<PayloadEntry> > chain_map_entries;
  // Carrier id of every source cell, degree-major; -1 for an empty carrier
  // and, on empty_carrier and not_acyclic, from the failing cell on.
  std::vector<int64_t> carrier_ids;
};

namespace detail {

/// The target d-cells whose vertices all carry `stamp` in `member`, in
/// complex order; `vertices` are the sorted vertex rows of the carrier.
inline void InducedCells ( const SimplicialComplex & target,
                           int64_t d,
                           const int32_t * vertices_begin,
                           const int32_t * vertices_end,
                           const std::vector<int64_t> & member,
                           int64_t stamp,
                           std::vector<int32_t> & cells ) {
  cells . clear ();
  if ( d >= target . degrees () ) return;
  if ( d == 0 ) {
    cells . assign ( vertices_begin, vertices_end );
    return;
  }
  const std::vector<int64_t> & first = target . first [ d ];
  for ( const int32_t * v = vertices_begin; v != vertices_end; ++ v ) {
    for ( int64_t row = first [ * v ]; row < first [ * v + 1 ]; ++ row ) {
      const int32_t * cell = target . cell ( d, row );
      bool inside = true;
      for ( int64_t k = 1; k <= d; ++ k ) {
        if ( member [ cell [ k ] ] != stamp ) {
          inside = false;
          break;
        }
      }
      if ( inside ) cells . push_back ( static_cast<int32_t> ( row ) );
    }
  }
}

inline void SetLocalRows ( const std::vector<int32_t> & rows,
                           std::vector<int32_t> & local ) {
  for ( size_t i = 0; i < rows . size (); ++ i ) {
    local [ rows [ i ] ] = static_cast<int32_t> ( i );
  }
}

inline void ClearLocalRows ( const std::vector<int32_t> & rows,
                             std::vector<int32_t> & local ) {
  for ( size_t i = 0; i < rows . size (); ++ i ) local [ rows [ i ] ] = -1;
}

/// Reduce the boundary columns of the carrier's d-cells; the local rows of
/// its (d-1)-cells must be set in `local`.
inline void EliminateCarrierBoundary ( const SimplicialComplex & target,
                                       int64_t d,
                                       const std::vector<int32_t> & rows,
                                       const std::vector<int32_t> & columns,
                                       const std::vector<int32_t> & local,
                                       bool combinations,
                                       ColumnEliminator & eliminator,
                                       std::vector<int32_t> & scratch ) {
  eliminator . reset ( static_cast<int64_t> ( rows . size () ),
                       static_cast<int64_t> ( columns . size () ),
                       combinations );
  const int64_t width = d + 1;
  scratch . resize ( width );
  for ( size_t j = 0; j < columns . size (); ++ j ) {
    const int32_t * faces = target . face_rows ( d, columns [ j ] );
    for ( int64_t i = 0; i < width; ++ i ) {
      const int32_t row = local [ faces [ i ] ];
      if ( row < 0 ) {
        throw std::runtime_error (
          "internal error: an induced carrier is not closed under faces" );
      }
      scratch [ i ] = row;
    }
    eliminator . add_column ( static_cast<int32_t> ( j ), scratch . data (), width );
  }
}

inline bool CellInExitSet ( const SimplicialComplex & complex,
                            const std::vector<uint8_t> & exit,
                            int64_t d,
                            int64_t row ) {
  const int32_t * cell = complex . cell ( d, row );
  for ( int64_t k = 0; k <= d; ++ k ) {
    if ( exit [ cell [ k ] ] == 0 ) return false;
  }
  return true;
}

inline void CheckExitMask ( const std::vector<uint8_t> & mask,
                            int64_t vertices,
                            const std::string & name,
                            const std::string & complex_name ) {
  if ( static_cast<int64_t> ( mask . size () ) != vertices ) {
    std::ostringstream message;
    message << name << " must have one entry per row of " << complex_name
            << "[0] (" << vertices << "); got " << mask . size ();
    throw std::invalid_argument ( message . str () );
  }
  for ( size_t i = 0; i < mask . size (); ++ i ) {
    if ( mask [ i ] > 1 ) {
      std::ostringstream message;
      message << name << "[" << i << "] must be 0 or 1";
      throw std::invalid_argument ( message . str () );
    }
  }
}

} // namespace detail

/// Construct and validate the chain map induced by the carrier of a vertex
/// map; see the comment at the top of this header.
inline CarrierChainMapResult
ComputeCarrierChainMap ( const CarrierChainMapInput & input ) {
  CarrierChainMapResult result;

  SimplicialComplex source;
  BuildSimplicialComplex ( input . source_simplices, "source_simplices", source );
  SimplicialComplex target_storage;
  if ( ! input . target_is_source ) {
    BuildSimplicialComplex (
      input . target_simplices, "target_simplices", target_storage );
  }
  const SimplicialComplex & target =
    input . target_is_source ? source : target_storage;
  const std::string target_name =
    input . target_is_source ? "source_simplices" : "target_simplices";

  const int64_t source_degrees = source . degrees ();
  const int64_t target_degrees = target . degrees ();
  const int64_t source_vertices = source . count ( 0 );
  const int64_t target_vertices = target . count ( 0 );

  detail::CheckExitMask (
    input . source_exit, source_vertices, "source_exit", "source_simplices" );
  if ( ! input . target_is_source ) {
    detail::CheckExitMask (
      input . target_exit, target_vertices, "target_exit", "target_simplices" );
  }
  const std::vector<uint8_t> & source_exit = input . source_exit;
  const std::vector<uint8_t> & target_exit =
    input . target_is_source ? input . source_exit : input . target_exit;

  // Vertex images, as sorted and distinct target vertex rows.
  const std::vector<int64_t> & indptr = input . vertex_image_indptr;
  const std::vector<int32_t> & indices = input . vertex_image_indices;
  const int64_t index_count = static_cast<int64_t> ( indices . size () );
  if ( static_cast<int64_t> ( indptr . size () ) != source_vertices + 1 ) {
    std::ostringstream message;
    message << "vertex_image_indptr must have length len(source_simplices[0]) + 1 = "
            << source_vertices + 1 << "; got " << indptr . size ();
    throw std::invalid_argument ( message . str () );
  }
  for ( int64_t i = 0; i <= source_vertices; ++ i ) {
    if ( indptr [ i ] < 0 || indptr [ i ] > index_count ) {
      std::ostringstream message;
      message << "vertex_image_indptr[" << i << "] = " << indptr [ i ]
              << " is outside [0, len(vertex_image_indices)] = [0, "
              << index_count << "]";
      throw std::out_of_range ( message . str () );
    }
  }
  if ( indptr [ 0 ] != 0 ) {
    throw std::invalid_argument ( "vertex_image_indptr[0] must be 0" );
  }
  for ( int64_t i = 0; i < source_vertices; ++ i ) {
    if ( indptr [ i + 1 ] < indptr [ i ] ) {
      std::ostringstream message;
      message << "vertex_image_indptr decreases from position " << i
              << " to position " << i + 1;
      throw std::invalid_argument ( message . str () );
    }
  }
  if ( indptr [ source_vertices ] != index_count ) {
    std::ostringstream message;
    message << "vertex_image_indptr[" << source_vertices << "] = "
            << indptr [ source_vertices ]
            << " must equal len(vertex_image_indices) = " << index_count;
    throw std::invalid_argument ( message . str () );
  }
  std::vector<int64_t> image_begin ( source_vertices + 1, 0 );
  std::vector<int32_t> image_rows;
  image_rows . reserve ( indices . size () );
  for ( int64_t v = 0; v < source_vertices; ++ v ) {
    const size_t start = image_rows . size ();
    for ( int64_t e = indptr [ v ]; e < indptr [ v + 1 ]; ++ e ) {
      const int32_t label = indices [ e ];
      const int32_t vertex = target . vertex_row ( label );
      if ( vertex < 0 ) {
        std::ostringstream message;
        message << "vertex_image_indices[" << e << "] = " << label
                << " (in the image of source vertex row " << v
                << ") is not a vertex label of " << target_name;
        throw std::invalid_argument ( message . str () );
      }
      image_rows . push_back ( vertex );
    }
    std::sort ( image_rows . begin () + start, image_rows . end () );
    image_rows . erase (
      std::unique ( image_rows . begin () + start, image_rows . end () ),
      image_rows . end () );
    image_begin [ v + 1 ] = static_cast<int64_t> ( image_rows . size () );
  }

  // Carriers of the source cells, interned in complex order, so that the
  // distinct carriers are numbered in order of first use.
  CarrierTable carriers;
  std::vector<std::vector<int64_t> > carrier_of ( source_degrees );
  for ( int64_t d = 0; d < source_degrees; ++ d ) {
    carrier_of [ d ] . assign ( source . count ( d ), -1 );
  }
  std::vector<int64_t> member ( target_vertices, -1 );
  int64_t stamp = 0;
  std::vector<int32_t> vertices;

  // Record the carrier numbers of the source cells, -1 for a cell whose
  // carrier was not interned, and the number of distinct carriers.
  auto record_carriers = [ & ] ( int64_t carrier_count ) {
    result . carrier_count = carrier_count;
    for ( int64_t d = 0; d < source_degrees; ++ d ) {
      result . carrier_ids . insert ( result . carrier_ids . end (),
                                     carrier_of [ d ] . begin (),
                                     carrier_of [ d ] . end () );
    }
  };

  // Empty carriers. The carrier of a cell is empty exactly when every vertex
  // of the cell has an empty image, and the vertices of a cell are 0-cells,
  // which come first in complex order; so the first source cell with an empty
  // carrier, if there is one, is the first vertex with an empty image. The
  // carriers of the vertices before it, which are their images, are recorded.
  int64_t empty_row = -1;
  for ( int64_t v = 0; v < source_vertices && empty_row < 0; ++ v ) {
    if ( image_begin [ v ] == image_begin [ v + 1 ] ) empty_row = v;
  }
  if ( empty_row >= 0 ) {
    bool inserted = false;
    for ( int64_t row = 0; row < empty_row; ++ row ) {
      vertices . assign ( image_rows . begin () + image_begin [ row ],
                          image_rows . begin () + image_begin [ row + 1 ] );
      carrier_of [ 0 ] [ row ] = carriers . intern ( vertices, inserted );
    }
    record_carriers ( carriers . count () );
    result . status = "empty_carrier";
    result . failure_degree = 0;
    result . failure_row = empty_row;
    return result;
  }

  // Mark the vertices of a carrier in `member` under a fresh stamp.
  auto mark_carrier = [ & ] ( int64_t id ) {
    ++ stamp;
    for ( const int32_t * v = carriers . begin ( id ); v != carriers . end ( id ); ++ v ) {
      member [ * v ] = stamp;
    }
    return stamp;
  };

  ColumnEliminator eliminator;
  std::vector<std::vector<int32_t> > local ( target_degrees );
  for ( int64_t d = 0; d < target_degrees; ++ d ) {
    local [ d ] . assign ( target . count ( d ), -1 );
  }
  std::vector<int32_t> scratch;

  // Whether a carrier is acyclic over GF(5): its Betti numbers are
  // ( 1, 0, ..., 0 ).
  std::vector<std::vector<int32_t> > carrier_cells ( target_degrees );
  std::vector<int64_t> ranks ( target_degrees + 1, 0 );
  auto acyclic = [ & ] ( int64_t id ) {
    const int64_t current = mark_carrier ( id );
    for ( int64_t d = 0; d < target_degrees; ++ d ) {
      detail::InducedCells ( target, d, carriers . begin ( id ), carriers . end ( id ),
                             member, current, carrier_cells [ d ] );
    }
    std::fill ( ranks . begin (), ranks . end (), 0 );
    for ( int64_t d = 1; d < target_degrees; ++ d ) {
      if ( carrier_cells [ d ] . empty () ) continue;
      detail::SetLocalRows ( carrier_cells [ d - 1 ], local [ d - 1 ] );
      detail::EliminateCarrierBoundary ( target, d, carrier_cells [ d - 1 ],
                                         carrier_cells [ d ], local [ d - 1 ], false,
                                         eliminator, scratch );
      detail::ClearLocalRows ( carrier_cells [ d - 1 ], local [ d - 1 ] );
      ranks [ d ] = eliminator . rank ();
    }
    for ( int64_t d = 0; d < target_degrees; ++ d ) {
      const int64_t betti = static_cast<int64_t> ( carrier_cells [ d ] . size () )
                            - ranks [ d ] - ranks [ d + 1 ];
      if ( betti != ( d == 0 ? 1 : 0 ) ) return false;
    }
    return true;
  };

  // Acyclicity, block by block of source cells in complex order. The
  // carriers of a block are interned first, and the new ones are then
  // checked in order of first use, so that every distinct carrier is checked
  // once. The first cell whose carrier is not acyclic is the first use of
  // that carrier, and the check stops there: the carriers of the cells
  // before it are recorded, the failing carrier is not counted, and the
  // cells after it in the block lose their carrier numbers again. The block
  // limits double from one block to the next: a successful call alternates
  // between the two steps only a few times, and a call that fails early
  // interns few cells past the failing one.
  {
    // First use of every new carrier of a block, as ( degree << 32 ) | row.
    std::vector<int64_t> first_use;
    const int64_t largest = std::numeric_limits<int64_t>::max () / 2;
    int64_t block_cells = std::min ( largest, std::max ( int64_t ( 1 ), input . block_cells ) );
    int64_t block_vertices =
      std::min ( largest, std::max ( int64_t ( 1 ), input . block_vertices ) );
    int64_t d = 0;
    int64_t row = 0;
    while ( d < source_degrees ) {
      first_use . clear ();
      const int64_t first_new = carriers . count ();
      int64_t interned_cells = 0;
      int64_t interned_vertices = 0;
      while ( d < source_degrees && interned_cells < block_cells &&
              interned_vertices < block_vertices ) {
        if ( row == source . count ( d ) ) {
          ++ d;
          row = 0;
          continue;
        }
        // The sorted vertex rows of the carrier of the cell.
        const int32_t * cell = source . cell ( d, row );
        vertices . clear ();
        ++ stamp;
        for ( int64_t k = 0; k <= d; ++ k ) {
          for ( int64_t e = image_begin [ cell [ k ] ];
                e < image_begin [ cell [ k ] + 1 ]; ++ e ) {
            const int32_t vertex = image_rows [ e ];
            if ( member [ vertex ] != stamp ) {
              member [ vertex ] = stamp;
              vertices . push_back ( vertex );
            }
          }
        }
        if ( d > 0 ) std::sort ( vertices . begin (), vertices . end () );
        bool inserted = false;
        carrier_of [ d ] [ row ] = carriers . intern ( vertices, inserted );
        if ( inserted ) first_use . push_back ( ( d << 32 ) | row );
        ++ interned_cells;
        interned_vertices += static_cast<int64_t> ( vertices . size () );
        ++ row;
      }
      for ( size_t k = 0; k < first_use . size (); ++ k ) {
        const int64_t id = first_new + static_cast<int64_t> ( k );
        if ( acyclic ( id ) ) continue;
        const int64_t failure_degree = first_use [ k ] >> 32;
        const int64_t failure_row = first_use [ k ] & 0xFFFFFFFF;
        // Clear the cells from the failing one to the end of the block.
        for ( int64_t e = failure_degree, r = failure_row; e < d || ( e == d && r < row ); ) {
          if ( r == source . count ( e ) ) {
            ++ e;
            r = 0;
            continue;
          }
          carrier_of [ e ] [ r ++ ] = -1;
        }
        record_carriers ( id );
        result . status = "not_acyclic";
        result . failure_degree = failure_degree;
        result . failure_row = failure_row;
        return result;
      }
      if ( block_cells < largest ) block_cells *= 2;
      if ( block_vertices < largest ) block_vertices *= 2;
    }
  }
  record_carriers ( carriers . count () );

  // Pair preservation of the carrier: a cell of the source P0 has its carrier
  // in the target P0, that is, its carrier vertices are exit vertices.
  for ( int64_t d = 0; d < source_degrees; ++ d ) {
    for ( int64_t row = 0; row < source . count ( d ); ++ row ) {
      if ( ! detail::CellInExitSet ( source, source_exit, d, row ) ) continue;
      const int64_t id = carrier_of [ d ] [ row ];
      for ( const int32_t * v = carriers . begin ( id ); v != carriers . end ( id ); ++ v ) {
        if ( target_exit [ * v ] == 0 ) {
          result . status = "pair_violation";
          result . failure_degree = d;
          result . failure_row = row;
          return result;
        }
      }
    }
  }

  // The chain map, degree by degree, as CSR over the source rows.
  std::vector<std::vector<int64_t> > phi_begin ( source_degrees );
  std::vector<std::vector<int32_t> > phi_rows ( source_degrees );
  std::vector<std::vector<uint8_t> > phi_values ( source_degrees );

  phi_begin [ 0 ] . resize ( source_vertices + 1 );
  for ( int64_t row = 0; row < source_vertices; ++ row ) {
    phi_begin [ 0 ] [ row ] = row;
    phi_rows [ 0 ] . push_back ( * carriers . begin ( carrier_of [ 0 ] [ row ] ) );
    phi_values [ 0 ] . push_back ( 1 );
  }
  phi_begin [ 0 ] [ source_vertices ] = source_vertices;

  int64_t largest_target_count = 0;
  for ( int64_t d = 0; d < target_degrees; ++ d ) {
    largest_target_count = std::max ( largest_target_count, target . count ( d ) );
  }
  std::vector<uint8_t> accumulator ( largest_target_count, 0 );
  std::vector<int32_t> touched;

  {
    std::vector<int32_t> rows;
    std::vector<int32_t> columns;
    std::vector<std::pair<int32_t, uint8_t> > rhs;
    std::vector<std::pair<int32_t, uint8_t> > solution;
    for ( int64_t d = 1; d < source_degrees; ++ d ) {
      const int64_t count = source . count ( d );
      const std::vector<int64_t> & ids = carrier_of [ d ];
      std::vector<int32_t> order ( count );
      for ( int64_t row = 0; row < count; ++ row ) order [ row ] = static_cast<int32_t> ( row );
      std::stable_sort ( order . begin (), order . end (),
                         [ & ] ( int32_t a, int32_t b ) { return ids [ a ] < ids [ b ]; } );

      std::vector<int64_t> image_start ( count, 0 );
      std::vector<int64_t> image_length ( count, 0 );
      std::vector<int32_t> pool_rows;
      std::vector<uint8_t> pool_values;
      int64_t failure_row = -1;
      std::string failure_status;
      const bool rows_exist = d - 1 < target_degrees;

      for ( int64_t a = 0; a < count; ) {
        const int64_t id = ids [ order [ a ] ];
        int64_t b = a;
        while ( b < count && ids [ order [ b ] ] == id ) ++ b;

        const int64_t current = mark_carrier ( id );
        detail::InducedCells ( target, d - 1, carriers . begin ( id ), carriers . end ( id ),
                               member, current, rows );
        detail::InducedCells ( target, d, carriers . begin ( id ), carriers . end ( id ),
                               member, current, columns );
        if ( rows_exist ) detail::SetLocalRows ( rows, local [ d - 1 ] );
        if ( d < target_degrees ) {
          detail::EliminateCarrierBoundary ( target, d, rows, columns, local [ d - 1 ],
                                             true, eliminator, scratch );
        } else {
          eliminator . reset ( static_cast<int64_t> ( rows . size () ), 0, true );
        }

        for ( int64_t i = a; i < b; ++ i ) {
          const int64_t row = order [ i ];
          // rhs = sum over faces of (-1)^i phi ( face ), on target (d-1)-rows.
          touched . clear ();
          const int32_t * faces = source . face_rows ( d, row );
          for ( int64_t k = 0; k <= d; ++ k ) {
            const uint8_t incidence = Incidence5 ( k );
            const int64_t face = faces [ k ];
            for ( int64_t e = phi_begin [ d - 1 ] [ face ];
                  e < phi_begin [ d - 1 ] [ face + 1 ]; ++ e ) {
              const int32_t t = phi_rows [ d - 1 ] [ e ];
              if ( accumulator [ t ] == 0 ) touched . push_back ( t );
              accumulator [ t ] = Add5 ( accumulator [ t ],
                                         Multiply5 ( incidence, phi_values [ d - 1 ] [ e ] ) );
            }
          }
          rhs . clear ();
          bool outside = false;
          for ( size_t e = 0; e < touched . size (); ++ e ) {
            const int32_t t = touched [ e ];
            const uint8_t value = accumulator [ t ];
            accumulator [ t ] = 0;
            if ( value == 0 ) continue;
            const int32_t local_row = rows_exist ? local [ d - 1 ] [ t ] : -1;
            if ( local_row < 0 ) {
              outside = true;
            } else {
              rhs . push_back ( std::make_pair ( local_row, value ) );
            }
          }
          if ( outside || ! eliminator . solve ( rhs, solution ) ) {
            if ( failure_row < 0 || row < failure_row ) {
              failure_row = row;
              failure_status = outside ? "chain_map_invalid" : "no_solution";
            }
            continue;
          }
          image_start [ row ] = static_cast<int64_t> ( pool_rows . size () );
          image_length [ row ] = static_cast<int64_t> ( solution . size () );
          for ( size_t e = 0; e < solution . size (); ++ e ) {
            pool_rows . push_back ( columns [ solution [ e ] . first ] );
            pool_values . push_back ( solution [ e ] . second );
          }
        }
        if ( rows_exist ) detail::ClearLocalRows ( rows, local [ d - 1 ] );
        a = b;
      }

      if ( failure_row >= 0 ) {
        result . status = failure_status;
        result . failure_degree = d;
        result . failure_row = failure_row;
        return result;
      }

      phi_begin [ d ] . resize ( count + 1 );
      phi_rows [ d ] . reserve ( pool_rows . size () );
      phi_values [ d ] . reserve ( pool_values . size () );
      for ( int64_t row = 0; row < count; ++ row ) {
        phi_begin [ d ] [ row ] = static_cast<int64_t> ( phi_rows [ d ] . size () );
        for ( int64_t e = image_start [ row ];
              e < image_start [ row ] + image_length [ row ]; ++ e ) {
          phi_rows [ d ] . push_back ( pool_rows [ e ] );
          phi_values [ d ] . push_back ( pool_values [ e ] );
        }
      }
      phi_begin [ d ] [ count ] = static_cast<int64_t> ( phi_rows [ d ] . size () );
    }
  }

  // Validation, in the order of hybrid_dynamics: the chain-map equation on
  // every cell, then the carriers, then the pair.
  auto invalid = [ & ] ( int64_t d, int64_t row ) {
    result . status = "chain_map_invalid";
    result . failure_degree = d;
    result . failure_row = row;
    return result;
  };
  for ( int64_t d = 0; d < source_degrees; ++ d ) {
    for ( int64_t row = 0; row < source . count ( d ); ++ row ) {
      for ( int64_t e = phi_begin [ d ] [ row ]; e < phi_begin [ d ] [ row + 1 ]; ++ e ) {
        const uint8_t value = phi_values [ d ] [ e ];
        if ( value == 0 || value > 4 || phi_rows [ d ] [ e ] < 0 ||
             phi_rows [ d ] [ e ] >= target . count ( d ) ) {
          return invalid ( d, row );
        }
      }
      if ( d == 0 ) continue;
      // phi ( boundary ( s ) ) - boundary ( phi ( s ) ) must vanish.
      touched . clear ();
      auto add = [ & ] ( int32_t t, uint8_t value ) {
        if ( accumulator [ t ] == 0 ) touched . push_back ( t );
        accumulator [ t ] = Add5 ( accumulator [ t ], value );
      };
      const int32_t * faces = source . face_rows ( d, row );
      for ( int64_t k = 0; k <= d; ++ k ) {
        const int64_t face = faces [ k ];
        for ( int64_t e = phi_begin [ d - 1 ] [ face ];
              e < phi_begin [ d - 1 ] [ face + 1 ]; ++ e ) {
          add ( phi_rows [ d - 1 ] [ e ],
                Multiply5 ( Incidence5 ( k ), phi_values [ d - 1 ] [ e ] ) );
        }
      }
      for ( int64_t e = phi_begin [ d ] [ row ]; e < phi_begin [ d ] [ row + 1 ]; ++ e ) {
        const int32_t * target_faces = target . face_rows ( d, phi_rows [ d ] [ e ] );
        for ( int64_t k = 0; k <= d; ++ k ) {
          add ( target_faces [ k ],
                Negate5 ( Multiply5 ( phi_values [ d ] [ e ], Incidence5 ( k ) ) ) );
        }
      }
      bool zero = true;
      for ( size_t e = 0; e < touched . size (); ++ e ) {
        if ( accumulator [ touched [ e ] ] != 0 ) zero = false;
        accumulator [ touched [ e ] ] = 0;
      }
      if ( ! zero ) return invalid ( d, row );
    }
  }
  for ( int64_t d = 0; d < source_degrees; ++ d ) {
    int64_t marked = -1;
    int64_t current = 0;
    for ( int64_t row = 0; row < source . count ( d ); ++ row ) {
      if ( carrier_of [ d ] [ row ] != marked ) {
        marked = carrier_of [ d ] [ row ];
        current = mark_carrier ( marked );
      }
      for ( int64_t e = phi_begin [ d ] [ row ]; e < phi_begin [ d ] [ row + 1 ]; ++ e ) {
        const int32_t * cell = target . cell ( d, phi_rows [ d ] [ e ] );
        for ( int64_t k = 0; k <= d; ++ k ) {
          if ( member [ cell [ k ] ] != current ) return invalid ( d, row );
        }
      }
    }
  }
  for ( int64_t d = 0; d < source_degrees; ++ d ) {
    for ( int64_t row = 0; row < source . count ( d ); ++ row ) {
      if ( ! detail::CellInExitSet ( source, source_exit, d, row ) ) continue;
      for ( int64_t e = phi_begin [ d ] [ row ]; e < phi_begin [ d ] [ row + 1 ]; ++ e ) {
        if ( ! detail::CellInExitSet ( target, target_exit, d, phi_rows [ d ] [ e ] ) ) {
          return invalid ( d, row );
        }
      }
    }
  }

  // Output in full complex indexing, unless it is not wanted.
  if ( input . return_chain_map ) {
    result . has_chain_map = true;
    result . chain_map . assign ( source_degrees, std::vector<int64_t> () );
    for ( int64_t d = 0; d < source_degrees; ++ d ) {
      std::vector<int64_t> & entries = result . chain_map [ d ];
      entries . reserve ( 3 * phi_rows [ d ] . size () );
      for ( int64_t row = 0; row < source . count ( d ); ++ row ) {
        for ( int64_t e = phi_begin [ d ] [ row ]; e < phi_begin [ d ] [ row + 1 ]; ++ e ) {
          entries . push_back ( row );
          entries . push_back ( phi_rows [ d ] [ e ] );
          entries . push_back ( phi_values [ d ] [ e ] );
        }
      }
    }
  }

  // The relative complex C(X) / C(P0) and the chain map on it, in the format
  // and entry order of hybrid_dynamics' to_cmgdb_payload: the basis of degree
  // d is the d-cells outside P0 in complex order; boundary entries are listed
  // by column and removal index, chain-map entries by column and in the order
  // of the image; entries at cells of P0 are dropped.
  if ( input . target_is_source ) {
    result . has_payload = true;
    std::vector<std::vector<int64_t> > basis ( source_degrees );
    result . cell_counts . assign ( source_degrees, 0 );
    for ( int64_t d = 0; d < source_degrees; ++ d ) {
      basis [ d ] . assign ( source . count ( d ), -1 );
      int64_t next = 0;
      for ( int64_t row = 0; row < source . count ( d ); ++ row ) {
        if ( ! detail::CellInExitSet ( source, source_exit, d, row ) ) {
          basis [ d ] [ row ] = next ++;
        }
      }
      result . cell_counts [ d ] = static_cast<uint64_t> ( next );
    }
    result . boundary_entries . assign ( source_degrees, std::vector<PayloadEntry> () );
    result . chain_map_entries . assign ( source_degrees, std::vector<PayloadEntry> () );
    for ( int64_t d = 0; d < source_degrees; ++ d ) {
      for ( int64_t row = 0; row < source . count ( d ); ++ row ) {
        const int64_t column = basis [ d ] [ row ];
        if ( column < 0 ) continue;
        if ( d > 0 ) {
          const int32_t * faces = source . face_rows ( d, row );
          for ( int64_t k = 0; k <= d; ++ k ) {
            const int64_t face = basis [ d - 1 ] [ faces [ k ] ];
            if ( face < 0 ) continue;
            result . boundary_entries [ d ] . push_back ( PayloadEntry (
              static_cast<uint64_t> ( face ), static_cast<uint64_t> ( column ),
              static_cast<int> ( Incidence5 ( k ) ) ) );
          }
        }
        for ( int64_t e = phi_begin [ d ] [ row ]; e < phi_begin [ d ] [ row + 1 ]; ++ e ) {
          const int64_t image = basis [ d ] [ phi_rows [ d ] [ e ] ];
          if ( image < 0 ) continue;
          result . chain_map_entries [ d ] . push_back ( PayloadEntry (
            static_cast<uint64_t> ( image ), static_cast<uint64_t> ( column ),
            static_cast<int> ( phi_values [ d ] [ e ] ) ) );
        }
      }
    }
  }
  return result;
}

} // namespace carrier_chain_map

#endif
