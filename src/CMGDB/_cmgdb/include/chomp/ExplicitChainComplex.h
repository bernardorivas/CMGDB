// ExplicitChainComplex.h
//
// A finite based chain complex supplied by explicit sparse boundary matrices.
// This is deliberately independent of CubicalComplex: callers with cellular
// suspension quotients should not have to encode their cells as fake cubes.

#ifndef CHOMP_EXPLICITCHAINCOMPLEX_H
#define CHOMP_EXPLICITCHAINCOMPLEX_H

#include <vector>

#include "chomp/Complex.h"
#include "chomp/SparseMatrix.h"

namespace chomp {

class ExplicitChainComplex : public Complex {
public:
  typedef SparseMatrix<Ring> Matrix;

  ExplicitChainComplex ( const std::vector<uint64_t> & cell_counts,
                         const std::vector<Matrix> & boundaries )
      : boundaries_ ( boundaries ) {
    dimension_ = static_cast<int> ( cell_counts . size () ) - 1;
    sizes_ . assign ( cell_counts . begin (), cell_counts . end () );
  }

  virtual void boundary ( Chain * output, const Index input, int dim ) const {
    output -> dimension () = dim - 1;
    if ( dim <= 0 || dim > dimension_ ) return;

    const Matrix & matrix = boundaries_ [ dim ];
    for ( Matrix::MatrixPosition entry = matrix . column_begin ( input );
          entry != matrix . end ();
          matrix . column_advance ( entry ) ) {
      (*output) += Term (
        static_cast<Index> ( matrix . row ( entry ) ), matrix . read ( entry ) );
    }
  }

  virtual void coboundary ( Chain * output, const Index input, int dim ) const {
    output -> dimension () = dim + 1;
    if ( dim < 0 || dim >= dimension_ ) return;

    const Matrix & matrix = boundaries_ [ dim + 1 ];
    for ( Matrix::MatrixPosition entry = matrix . row_begin ( input );
          entry != matrix . end ();
          matrix . row_advance ( entry ) ) {
      (*output) += Term (
        static_cast<Index> ( matrix . column ( entry ) ), matrix . read ( entry ) );
    }
  }

private:
  std::vector<Matrix> boundaries_;
};

} // namespace chomp

#endif
