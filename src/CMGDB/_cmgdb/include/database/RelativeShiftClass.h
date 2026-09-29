// RelativeShiftClass.h
//
// Induced maps and shift classes of a chain endomorphism of a finite chain
// complex over GF(5), by plain linear algebra: the kernel of
// CMGDB.ComputeRelativeShiftClass.
//
// The input is that of ComputeRelativeHomologyShiftClass: the number of basis
// cells in every degree, the sparse boundary matrices, and the sparse matrices
// of the chain map. It is validated in the same order and with the same
// messages: bounds and duplicate coordinates, boundary^2 = 0, and the
// chain-map equation.
//
// Homology. The boundary matrices are column reduced from the top degree
// down: R_d = boundary_d * V_d with V_d upper unitriangular, where a column is
// reduced by cancelling its pivot (its largest nonzero row) against the
// earlier column with the same pivot. The columns of degree d that are pivots
// of R_{d+1} reduce to zero and are skipped (clearing). A column j of degree d
// that is neither skipped nor a pivot column of R_d is essential, and its
// column of V_d is a cycle z_j with pivot j. The nonzero columns of R_{d+1}
// (a basis of the boundaries) and the cycles z_j have distinct pivots, and
// together they are a basis of the cycles; so the classes of the z_j, in
// increasing order of j, are a basis of H_d.
//
// Induced map. The image of z_j is a cycle. Its coordinates in that basis are
// found by cancelling its pivot, against a column of R_{d+1} or against an
// essential cycle z_i (whose coefficient is recorded), until nothing is left.
// Column k of the induced matrix holds the coordinates of the image of the
// k-th basis cycle.
//
// Shift class. The invariant factors of an induced matrix A are the diagonal
// of the Smith form of xI - A over GF(5)[x], computed by Euclidean elimination
// with a pivot of least degree. Each factor is divided by its largest power
// of x, and the factors of positive degree that remain are written in
// divisibility order in the format of conleyIndexString ("0" when there are
// none).
//
// Every loop terminates: a reduction strictly decreases the pivot of the
// column being reduced, and the Smith form strictly decreases the degree of
// its pivot. CHomP's Morse reduction and Smith normal form are not used.

#ifndef CMDB_RELATIVE_SHIFT_CLASS_H
#define CMDB_RELATIVE_SHIFT_CLASS_H

#include <stdint.h>
#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace relative_shift_class {

typedef std::tuple<uint64_t, uint64_t, int> SparseEntry;
typedef std::vector<std::vector<SparseEntry> > GradedSparseEntries;

/// Arithmetic of GF(5) on the residues 0..4 stored as uint8_t.
inline uint8_t Residue5 ( int64_t value ) {
  const int64_t residue = value % 5;
  return static_cast<uint8_t> ( residue < 0 ? residue + 5 : residue );
}

inline uint8_t Add5 ( uint8_t a, uint8_t b ) {
  const uint8_t sum = static_cast<uint8_t> ( a + b );
  return sum >= 5 ? static_cast<uint8_t> ( sum - 5 ) : sum;
}

inline uint8_t Negate5 ( uint8_t a ) {
  return a == 0 ? 0 : static_cast<uint8_t> ( 5 - a );
}

inline uint8_t Multiply5 ( uint8_t a, uint8_t b ) {
  static const uint8_t table [ 5 ] [ 5 ] = {
    { 0, 0, 0, 0, 0 },
    { 0, 1, 2, 3, 4 },
    { 0, 2, 4, 1, 3 },
    { 0, 3, 1, 4, 2 },
    { 0, 4, 3, 2, 1 } };
  return table [ a ] [ b ];
}

inline uint8_t Inverse5 ( uint8_t a ) {
  static const uint8_t table [ 5 ] = { 0, 1, 3, 2, 4 };
  return table [ a ];
}

/// The representative in -2..2, as chomp::Zp<5>::balanced_value.
inline int64_t Balanced5 ( uint8_t a ) {
  return a > 2 ? static_cast<int64_t> ( a ) - 5 : static_cast<int64_t> ( a );
}

// ---------------------------------------------------------------------------
// Sparse chains
// ---------------------------------------------------------------------------

/// A sparse chain: packed entries ( index << 3 ) | value with nonzero values,
/// in increasing order of index. Its pivot is the index of the last entry.
typedef std::vector<uint64_t> Chain;

/// Indices must be below this bound to be packed.
const uint64_t kIndexLimit = uint64_t ( 1 ) << 60;

inline uint64_t Pack ( uint64_t index, uint8_t value ) {
  return ( index << 3 ) | value;
}

inline uint64_t IndexOf ( uint64_t entry ) {
  return entry >> 3;
}

inline uint8_t ValueOf ( uint64_t entry ) {
  return static_cast<uint8_t> ( entry & 7 );
}

/// target += factor * source, with scratch as the merge buffer.
inline void AddMultiple ( Chain * target,
                          const Chain & source,
                          uint8_t factor,
                          Chain * scratch ) {
  if ( factor == 0 || source . empty () ) return;
  scratch -> clear ();
  scratch -> reserve ( target -> size () + source . size () );
  Chain::const_iterator a = target -> begin ();
  Chain::const_iterator b = source . begin ();
  while ( a != target -> end () && b != source . end () ) {
    const uint64_t a_index = IndexOf ( * a );
    const uint64_t b_index = IndexOf ( * b );
    if ( a_index < b_index ) {
      scratch -> push_back ( * a );
      ++ a;
    } else if ( b_index < a_index ) {
      scratch -> push_back ( Pack ( b_index, Multiply5 ( factor, ValueOf ( * b ) ) ) );
      ++ b;
    } else {
      const uint8_t value =
        Add5 ( ValueOf ( * a ), Multiply5 ( factor, ValueOf ( * b ) ) );
      if ( value != 0 ) scratch -> push_back ( Pack ( a_index, value ) );
      ++ a;
      ++ b;
    }
  }
  for ( ; a != target -> end (); ++ a ) scratch -> push_back ( * a );
  for ( ; b != source . end (); ++ b ) {
    scratch -> push_back (
      Pack ( IndexOf ( * b ), Multiply5 ( factor, ValueOf ( * b ) ) ) );
  }
  target -> swap ( * scratch );
}

/// target = factor * target.
inline void Scale ( Chain * target, uint8_t factor ) {
  for ( uint64_t & entry : * target ) {
    entry = Pack ( IndexOf ( entry ), Multiply5 ( factor, ValueOf ( entry ) ) );
  }
}

/// A dense accumulator of sparse sums over 0..size-1.
class Accumulator {
public:
  explicit Accumulator ( uint64_t size ) : value_ ( size, 0 ), marked_ ( size, 0 ) {}

  void add ( uint64_t index, uint8_t value ) {
    if ( value == 0 ) return;
    if ( ! marked_ [ index ] ) {
      marked_ [ index ] = 1;
      touched_ . push_back ( index );
    }
    value_ [ index ] = Add5 ( value_ [ index ], value );
  }

  /// Add factor times every entry of the chain.
  void add ( const uint64_t * begin, const uint64_t * end, uint8_t factor ) {
    if ( factor == 0 ) return;
    for ( const uint64_t * entry = begin; entry != end; ++ entry ) {
      add ( IndexOf ( * entry ), Multiply5 ( factor, ValueOf ( * entry ) ) );
    }
  }

  /// True if every accumulated value is zero; then clear.
  bool vanishes ( void ) {
    bool zero = true;
    for ( uint64_t index : touched_ ) {
      if ( value_ [ index ] != 0 ) zero = false;
    }
    clear ();
    return zero;
  }

  /// Move the accumulated sum into a chain; then clear.
  void extract ( Chain * output ) {
    output -> clear ();
    std::sort ( touched_ . begin (), touched_ . end () );
    for ( uint64_t index : touched_ ) {
      if ( value_ [ index ] != 0 ) output -> push_back ( Pack ( index, value_ [ index ] ) );
    }
    clear ();
  }

private:
  void clear ( void ) {
    for ( uint64_t index : touched_ ) {
      value_ [ index ] = 0;
      marked_ [ index ] = 0;
    }
    touched_ . clear ();
  }

  std::vector<uint8_t> value_;
  std::vector<uint8_t> marked_;
  std::vector<uint64_t> touched_;
};

/// A sparse matrix over GF(5) stored by columns; every column is a Chain of
/// row indices.
struct SparseColumns {
  uint64_t rows = 0;
  std::vector<uint64_t> offsets;   // columns + 1
  std::vector<uint64_t> entries;

  uint64_t columns ( void ) const {
    return offsets . empty () ? 0 : offsets . size () - 1;
  }

  const uint64_t * begin ( uint64_t column ) const {
    return entries . data () + offsets [ column ];
  }

  const uint64_t * end ( uint64_t column ) const {
    return entries . data () + offsets [ column + 1 ];
  }
};

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

/// Validate and store the matrices of one degree. The first entry, in input
/// order, that is out of bounds or repeats an earlier coordinate is reported,
/// with the messages of ComputeRelativeHomologyShiftClass.
inline SparseColumns BuildColumns ( uint64_t rows,
                                    uint64_t columns,
                                    const std::vector<SparseEntry> & entries,
                                    size_t degree,
                                    bool boundary_matrix ) {
  if ( rows >= kIndexLimit || columns >= kIndexLimit ) {
    throw std::overflow_error ( "chain group is too large for ComputeRelativeShiftClass" );
  }
  const char * kind = boundary_matrix ? "boundary" : "chain map";
  const size_t count = entries . size ();

  size_t out_of_bounds = count;
  for ( size_t position = 0; position < count; ++ position ) {
    if ( std::get<0> ( entries [ position ] ) >= rows ||
         std::get<1> ( entries [ position ] ) >= columns ) {
      out_of_bounds = position;
      break;
    }
  }

  // The entries before the first one out of bounds, grouped by column and
  // sorted by ( row, position ).
  SparseColumns matrix;
  matrix . rows = rows;
  matrix . offsets . assign ( columns + 1, 0 );
  for ( size_t position = 0; position < out_of_bounds; ++ position ) {
    ++ matrix . offsets [ std::get<1> ( entries [ position ] ) + 1 ];
  }
  for ( uint64_t column = 0; column < columns; ++ column ) {
    matrix . offsets [ column + 1 ] += matrix . offsets [ column ];
  }
  std::vector<std::pair<uint64_t, uint64_t> > sorted ( out_of_bounds );
  {
    std::vector<uint64_t> next ( matrix . offsets . begin (), matrix . offsets . end () - 1 );
    for ( size_t position = 0; position < out_of_bounds; ++ position ) {
      const uint64_t column = std::get<1> ( entries [ position ] );
      sorted [ next [ column ] ++ ] =
        std::make_pair ( std::get<0> ( entries [ position ] ), static_cast<uint64_t> ( position ) );
    }
  }
  size_t duplicate = count;
  for ( uint64_t column = 0; column < columns; ++ column ) {
    const uint64_t first = matrix . offsets [ column ];
    const uint64_t last = matrix . offsets [ column + 1 ];
    std::sort ( sorted . begin () + first, sorted . begin () + last );
    for ( uint64_t k = first + 1; k < last; ++ k ) {
      // The second occurrence of a coordinate is the first to repeat it.
      if ( sorted [ k ] . first == sorted [ k - 1 ] . first &&
           ( k == first + 1 || sorted [ k - 2 ] . first != sorted [ k ] . first ) &&
           sorted [ k ] . second < duplicate ) {
        duplicate = static_cast<size_t> ( sorted [ k ] . second );
      }
    }
  }
  if ( duplicate < out_of_bounds ) {
    std::ostringstream message;
    message << "duplicate " << kind << " entry ("
            << std::get<0> ( entries [ duplicate ] ) << ", "
            << std::get<1> ( entries [ duplicate ] ) << ") in dimension " << degree;
    throw std::invalid_argument ( message . str () );
  }
  if ( out_of_bounds < count ) {
    std::ostringstream message;
    message << kind << " entry (" << std::get<0> ( entries [ out_of_bounds ] ) << ", "
            << std::get<1> ( entries [ out_of_bounds ] ) << ") in dimension " << degree
            << " is outside its " << rows << " x " << columns << " matrix";
    throw std::out_of_range ( message . str () );
  }

  // Values reduced mod 5; zeros are dropped.
  matrix . entries . reserve ( count );
  uint64_t written = 0;
  for ( uint64_t column = 0; column < columns; ++ column ) {
    const uint64_t first = matrix . offsets [ column ];
    const uint64_t last = matrix . offsets [ column + 1 ];
    matrix . offsets [ column ] = written;
    for ( uint64_t k = first; k < last; ++ k ) {
      const uint8_t value = Residue5 ( std::get<2> ( entries [ sorted [ k ] . second ] ) );
      if ( value != 0 ) {
        matrix . entries . push_back ( Pack ( sorted [ k ] . first, value ) );
        ++ written;
      }
    }
  }
  matrix . offsets [ columns ] = written;
  return matrix;
}

inline std::vector<SparseColumns> BuildMatrices (
    const std::vector<uint64_t> & cell_counts,
    const GradedSparseEntries & entries,
    bool boundary_matrices ) {
  if ( entries . size () != cell_counts . size () ) {
    throw std::invalid_argument (
      boundary_matrices
        ? "boundary_entries must have one list for every chain dimension"
        : "chain_map_entries must have one list for every chain dimension" );
  }
  std::vector<SparseColumns> result;
  result . reserve ( cell_counts . size () );
  for ( size_t d = 0; d < cell_counts . size (); ++ d ) {
    const uint64_t rows = boundary_matrices
      ? ( d == 0 ? 0 : cell_counts [ d - 1 ] )
      : cell_counts [ d ];
    result . push_back (
      BuildColumns ( rows, cell_counts [ d ], entries [ d ], d, boundary_matrices ) );
  }
  return result;
}

/// boundary^2 = 0 and boundary * map = map * boundary over GF(5), checked in
/// the order and with the messages of ComputeRelativeHomologyShiftClass.
inline void ValidateChainData ( const std::vector<SparseColumns> & boundaries,
                                const std::vector<SparseColumns> & chain_map ) {
  for ( size_t d = 2; d < boundaries . size (); ++ d ) {
    const SparseColumns & outer = boundaries [ d - 1 ];
    const SparseColumns & inner = boundaries [ d ];
    Accumulator sum ( outer . rows );
    for ( uint64_t column = 0; column < inner . columns (); ++ column ) {
      for ( const uint64_t * entry = inner . begin ( column );
            entry != inner . end ( column ); ++ entry ) {
        const uint64_t middle = IndexOf ( * entry );
        sum . add ( outer . begin ( middle ), outer . end ( middle ), ValueOf ( * entry ) );
      }
      if ( ! sum . vanishes () ) {
        std::ostringstream message;
        message << "boundary squared is nonzero from dimension " << d
                << " to dimension " << d - 2 << " (coefficients are in F_5)";
        throw std::invalid_argument ( message . str () );
      }
    }
  }
  for ( size_t d = 1; d < boundaries . size (); ++ d ) {
    const SparseColumns & boundary = boundaries [ d ];
    const SparseColumns & map = chain_map [ d ];
    const SparseColumns & lower_map = chain_map [ d - 1 ];
    Accumulator difference ( boundary . rows );
    for ( uint64_t column = 0; column < boundary . columns (); ++ column ) {
      for ( const uint64_t * entry = map . begin ( column );
            entry != map . end ( column ); ++ entry ) {
        const uint64_t middle = IndexOf ( * entry );
        difference . add ( boundary . begin ( middle ), boundary . end ( middle ),
                           ValueOf ( * entry ) );
      }
      for ( const uint64_t * entry = boundary . begin ( column );
            entry != boundary . end ( column ); ++ entry ) {
        const uint64_t middle = IndexOf ( * entry );
        difference . add ( lower_map . begin ( middle ), lower_map . end ( middle ),
                           Negate5 ( ValueOf ( * entry ) ) );
      }
      if ( ! difference . vanishes () ) {
        std::ostringstream message;
        message << "chain-map equation fails in dimension " << d
                << ": boundary[d] * map[d] != map[d-1] * boundary[d] over F_5";
        throw std::invalid_argument ( message . str () );
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Homology and induced maps
// ---------------------------------------------------------------------------

/// The reduction of one boundary matrix, R = boundary * V.
struct Reduction {
  std::vector<int64_t> pivot_column;    // by row: the column of R with that pivot, or -1
  std::vector<Chain> reduced;           // the nonzero columns of R, pivot coefficient 1
};

/// The basis of H_d: the essential columns in increasing order and their
/// cycles, each with pivot coefficient 1.
struct HomologyBasis {
  std::vector<uint64_t> essential;
  std::vector<Chain> cycles;
  std::vector<int64_t> position;        // by cell of degree d: index in essential, or -1
};

/// Reduce the columns of boundary_d, skipping the pivots of R_{d+1}. The
/// columns that reduce to zero give the homology basis in degree d.
inline void ReduceBoundary ( const SparseColumns & boundary,
                             const std::vector<int64_t> & cleared,
                             Reduction * reduction,
                             HomologyBasis * basis ) {
  const uint64_t columns = boundary . columns ();
  reduction -> pivot_column . assign ( boundary . rows, -1 );
  reduction -> reduced . assign ( columns, Chain () );
  basis -> essential . clear ();
  basis -> cycles . clear ();
  basis -> position . assign ( columns, -1 );

  std::vector<Chain> transform ( columns );   // columns of V for the pivot columns
  Chain column;
  Chain combination;
  Chain scratch;
  for ( uint64_t j = 0; j < columns; ++ j ) {
    if ( cleared [ j ] >= 0 ) continue;
    column . assign ( boundary . begin ( j ), boundary . end ( j ) );
    combination . assign ( 1, Pack ( j, 1 ) );
    while ( ! column . empty () ) {
      const int64_t k = reduction -> pivot_column [ IndexOf ( column . back () ) ];
      if ( k < 0 ) break;
      const uint8_t factor = Negate5 ( ValueOf ( column . back () ) );
      AddMultiple ( & column, reduction -> reduced [ k ], factor, & scratch );
      AddMultiple ( & combination, transform [ k ], factor, & scratch );
    }
    if ( column . empty () ) {
      basis -> position [ j ] = static_cast<int64_t> ( basis -> essential . size () );
      basis -> essential . push_back ( j );
      basis -> cycles . push_back ( combination );
    } else {
      const uint8_t unit = Inverse5 ( ValueOf ( column . back () ) );
      Scale ( & column, unit );
      Scale ( & combination, unit );
      reduction -> pivot_column [ IndexOf ( column . back () ) ] = static_cast<int64_t> ( j );
      reduction -> reduced [ j ] . swap ( column );
      transform [ j ] . swap ( combination );
    }
  }
}

/// The induced matrix on H_d, entries in 0..4: column k holds the coordinates
/// of the class of map ( z_k ).
inline std::vector<std::vector<uint8_t> >
InducedMatrix ( const SparseColumns & map,
                const HomologyBasis & basis,
                const Reduction & above ) {
  const size_t size = basis . essential . size ();
  std::vector<std::vector<uint8_t> > matrix ( size, std::vector<uint8_t> ( size, 0 ) );
  if ( size == 0 ) return matrix;
  Accumulator sum ( map . rows );
  Chain image;
  Chain scratch;
  for ( size_t k = 0; k < size; ++ k ) {
    for ( uint64_t entry : basis . cycles [ k ] ) {
      const uint64_t cell = IndexOf ( entry );
      sum . add ( map . begin ( cell ), map . end ( cell ), ValueOf ( entry ) );
    }
    sum . extract ( & image );
    while ( ! image . empty () ) {
      const uint64_t pivot = IndexOf ( image . back () );
      const uint8_t factor = Negate5 ( ValueOf ( image . back () ) );
      const int64_t boundary_column =
        above . pivot_column . empty () ? -1 : above . pivot_column [ pivot ];
      if ( boundary_column >= 0 ) {
        AddMultiple ( & image, above . reduced [ boundary_column ], factor, & scratch );
        continue;
      }
      const int64_t i = basis . position [ pivot ];
      if ( i < 0 ) {
        throw std::runtime_error (
          "internal error: the image of a homology basis cycle is not a cycle" );
      }
      matrix [ i ] [ k ] = ValueOf ( image . back () );
      AddMultiple ( & image, basis . cycles [ i ], factor, & scratch );
    }
  }
  return matrix;
}

// ---------------------------------------------------------------------------
// Invariant factors and shift classes
// ---------------------------------------------------------------------------

/// A polynomial over GF(5): coefficients, constant term first, with no
/// trailing zeros (the zero polynomial is empty).
typedef std::vector<uint8_t> Polynomial;

inline void Trim ( Polynomial * p ) {
  while ( ! p -> empty () && p -> back () == 0 ) p -> pop_back ();
}

inline int64_t Degree ( const Polynomial & p ) {
  return static_cast<int64_t> ( p . size () ) - 1;
}

/// a -= q * b.
inline void SubtractProduct ( Polynomial * a, const Polynomial & q, const Polynomial & b ) {
  if ( q . empty () || b . empty () ) return;
  if ( a -> size () < q . size () + b . size () - 1 ) {
    a -> resize ( q . size () + b . size () - 1, 0 );
  }
  for ( size_t i = 0; i < q . size (); ++ i ) {
    if ( q [ i ] == 0 ) continue;
    const uint8_t factor = Negate5 ( q [ i ] );
    for ( size_t j = 0; j < b . size (); ++ j ) {
      ( * a ) [ i + j ] = Add5 ( ( * a ) [ i + j ], Multiply5 ( factor, b [ j ] ) );
    }
  }
  Trim ( a );
}

/// quotient and remainder of a by a nonzero b.
inline void Divide ( const Polynomial & a,
                     const Polynomial & b,
                     Polynomial * quotient,
                     Polynomial * remainder ) {
  * remainder = a;
  quotient -> clear ();
  const int64_t divisor_degree = Degree ( b );
  if ( Degree ( a ) < divisor_degree ) return;
  quotient -> assign ( Degree ( a ) - divisor_degree + 1, 0 );
  const uint8_t unit = Inverse5 ( b . back () );
  while ( Degree ( * remainder ) >= divisor_degree ) {
    const int64_t shift = Degree ( * remainder ) - divisor_degree;
    const uint8_t factor = Multiply5 ( remainder -> back (), unit );
    ( * quotient ) [ shift ] = factor;
    const uint8_t negated = Negate5 ( factor );
    for ( int64_t j = 0; j <= divisor_degree; ++ j ) {
      ( * remainder ) [ shift + j ] =
        Add5 ( ( * remainder ) [ shift + j ], Multiply5 ( negated, b [ j ] ) );
    }
    Trim ( remainder );
  }
  Trim ( quotient );
}

/// The invariant factors of a square matrix over GF(5), monic and in
/// divisibility order: the diagonal of the Smith form of xI - A over
/// GF(5)[x].
inline std::vector<Polynomial>
InvariantFactors ( const std::vector<std::vector<uint8_t> > & a ) {
  const size_t n = a . size ();
  std::vector<std::vector<Polynomial> > m ( n, std::vector<Polynomial> ( n ) );
  for ( size_t i = 0; i < n; ++ i ) {
    for ( size_t j = 0; j < n; ++ j ) {
      Polynomial & entry = m [ i ] [ j ];
      entry . assign ( i == j ? 2 : 1, 0 );
      entry [ 0 ] = Negate5 ( a [ i ] [ j ] );
      if ( i == j ) entry [ 1 ] = 1;
      Trim ( & entry );
    }
  }

  Polynomial quotient;
  Polynomial remainder;
  std::vector<Polynomial> factors;
  for ( size_t k = 0; k < n; ++ k ) {
    while ( true ) {
      // A nonzero entry of least degree, preferring the current pivot.
      size_t best_row = n;
      size_t best_column = n;
      int64_t best_degree = std::numeric_limits<int64_t>::max ();
      if ( ! m [ k ] [ k ] . empty () ) {
        best_row = k;
        best_column = k;
        best_degree = Degree ( m [ k ] [ k ] );
      }
      for ( size_t i = k; i < n && best_degree > 0; ++ i ) {
        for ( size_t j = k; j < n; ++ j ) {
          if ( ! m [ i ] [ j ] . empty () && Degree ( m [ i ] [ j ] ) < best_degree ) {
            best_row = i;
            best_column = j;
            best_degree = Degree ( m [ i ] [ j ] );
          }
        }
      }
      if ( best_row == n ) {
        throw std::runtime_error (
          "internal error: the characteristic matrix of an induced map is singular" );
      }
      if ( best_row != k ) m [ k ] . swap ( m [ best_row ] );
      if ( best_column != k ) {
        for ( size_t i = k; i < n; ++ i ) m [ i ] [ k ] . swap ( m [ i ] [ best_column ] );
      }

      // Divide the rest of column k and of row k by the pivot.
      bool remainders = false;
      for ( size_t i = k + 1; i < n; ++ i ) {
        if ( m [ i ] [ k ] . empty () ) continue;
        Divide ( m [ i ] [ k ], m [ k ] [ k ], & quotient, & remainder );
        for ( size_t j = k; j < n; ++ j ) SubtractProduct ( & m [ i ] [ j ], quotient, m [ k ] [ j ] );
        if ( ! m [ i ] [ k ] . empty () ) remainders = true;
      }
      for ( size_t j = k + 1; j < n; ++ j ) {
        if ( m [ k ] [ j ] . empty () ) continue;
        Divide ( m [ k ] [ j ], m [ k ] [ k ], & quotient, & remainder );
        for ( size_t i = k; i < n; ++ i ) SubtractProduct ( & m [ i ] [ j ], quotient, m [ i ] [ k ] );
        if ( ! m [ k ] [ j ] . empty () ) remainders = true;
      }
      if ( remainders ) continue;   // a remainder has smaller degree than the pivot

      // The pivot must divide every remaining entry; otherwise add that row
      // to row k, which leaves a remainder in row k.
      size_t offending = n;
      for ( size_t i = k + 1; i < n && offending == n; ++ i ) {
        for ( size_t j = k + 1; j < n; ++ j ) {
          if ( m [ i ] [ j ] . empty () ) continue;
          Divide ( m [ i ] [ j ], m [ k ] [ k ], & quotient, & remainder );
          if ( ! remainder . empty () ) {
            offending = i;
            break;
          }
        }
      }
      if ( offending == n ) break;
      for ( size_t j = k + 1; j < n; ++ j ) m [ k ] [ j ] = m [ offending ] [ j ];
    }
    Polynomial factor = m [ k ] [ k ];
    const uint8_t unit = Inverse5 ( factor . back () );
    for ( uint8_t & coefficient : factor ) coefficient = Multiply5 ( coefficient, unit );
    factors . push_back ( factor );
  }
  return factors;
}

/// A polynomial as chomp::PolyRing<Zp<5>> prints it.
inline std::string FormatPolynomial ( const Polynomial & p ) {
  if ( p . empty () ) return "0";
  std::ostringstream text;
  const int64_t degree = Degree ( p );
  for ( int64_t i = degree; i >= 0; -- i ) {
    if ( p [ i ] == 0 ) continue;
    const int64_t value = Balanced5 ( p [ i ] );
    if ( i != degree && value > 0 ) text << "+";
    if ( value == -1 ) text << "-";
    if ( value != 1 && value != -1 ) text << value;
    if ( i > 1 ) {
      text << "x^" << i;
    } else if ( i == 1 ) {
      text << "x";
    } else if ( value == 1 || value == -1 ) {
      text << "1";
    }
  }
  return text . str ();
}

/// The shift-class string of conleyIndexString: the invariant factors with
/// their powers of x removed, those of positive degree concatenated, or "0".
inline std::string ShiftClassString ( const std::vector<std::vector<uint8_t> > & matrix ) {
  std::string result;
  for ( const Polynomial & factor : InvariantFactors ( matrix ) ) {
    size_t zeros = 0;
    while ( zeros < factor . size () && factor [ zeros ] == 0 ) ++ zeros;
    const Polynomial stripped ( factor . begin () + zeros, factor . end () );
    if ( Degree ( stripped ) > 0 ) result += FormatPolynomial ( stripped );
  }
  return result . empty () ? std::string ( "0" ) : result;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

struct RelativeShiftClassResult {
  std::vector<std::string> shift_class;
  std::vector<uint64_t> homology_dimensions;
  std::vector<std::vector<std::vector<int64_t> > > induced_maps;
};

inline RelativeShiftClassResult
ComputeRelativeShiftClass ( const std::vector<uint64_t> & cell_counts,
                            const GradedSparseEntries & boundary_entries,
                            const GradedSparseEntries & chain_map_entries ) {
  if ( cell_counts . empty () ) {
    throw std::invalid_argument ( "cell_counts must contain at least dimension zero" );
  }
  const std::vector<SparseColumns> boundaries =
    BuildMatrices ( cell_counts, boundary_entries, true );
  const std::vector<SparseColumns> chain_map =
    BuildMatrices ( cell_counts, chain_map_entries, false );
  ValidateChainData ( boundaries, chain_map );

  const size_t degrees = cell_counts . size ();
  RelativeShiftClassResult result;
  result . shift_class . resize ( degrees );
  result . homology_dimensions . resize ( degrees );
  result . induced_maps . resize ( degrees );

  // From the top degree down, R_{d+1} is kept while degree d is computed.
  Reduction above;
  for ( size_t step = 0; step < degrees; ++ step ) {
    const size_t d = degrees - 1 - step;
    const uint64_t cells = cell_counts [ d ];
    std::vector<int64_t> cleared ( cells, -1 );
    if ( ! above . pivot_column . empty () ) cleared = above . pivot_column;

    Reduction reduction;
    HomologyBasis basis;
    ReduceBoundary ( boundaries [ d ], cleared, & reduction, & basis );
    const std::vector<std::vector<uint8_t> > matrix =
      InducedMatrix ( chain_map [ d ], basis, above );

    const size_t size = matrix . size ();
    result . homology_dimensions [ d ] = static_cast<uint64_t> ( size );
    std::vector<std::vector<int64_t> > dense ( size, std::vector<int64_t> ( size, 0 ) );
    for ( size_t i = 0; i < size; ++ i ) {
      for ( size_t j = 0; j < size; ++ j ) dense [ i ] [ j ] = Balanced5 ( matrix [ i ] [ j ] );
    }
    result . induced_maps [ d ] = std::move ( dense );
    result . shift_class [ d ] = ShiftClassString ( matrix );
    above = std::move ( reduction );
  }
  return result;
}

} // namespace relative_shift_class

#endif
