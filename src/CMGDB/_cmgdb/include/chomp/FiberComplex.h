// FiberComplex.h
// Shaun Harker
// 9/16/11
///
/// Marcio Gameiro
/// 2025-06-15

#ifndef CHOMP_FIBERCOMPLEX_H
#define CHOMP_FIBERCOMPLEX_H

#include <cstdlib>
#include <stdexcept>
#include <vector>
#include "boost/unordered_set.hpp"
#include "boost/unordered_map.hpp"

#include "chomp/Complex.h"
#include "chomp/Chain.h"
#include "chomp/Preboundary.h"
#include "chomp/Closure.h"
#include "chomp/CubicalComplex.h"
#include "chomp/Generators.h"


// debug
//#include "chomp/Draw.h"

namespace chomp { 
  
class FiberComplex : public Complex {
public:
CHOMP_COMPLEX(Index)
private:
  CubicalComplex * supercomplex_;
  Index choose_ = 0;
  void buildForFilling ( const boost::unordered_set<Index>& X_nbs,
                         const boost::unordered_set<Index>& A_nbs,
                         const std::unordered_map<Index, std::vector<Index>>& images );
public:
  template < class RectMap >
  FiberComplex ( const boost::unordered_set < Index > & X_nbs,
                 const boost::unordered_set < Index > & A_nbs,
                 const CubicalComplex & domain,
                 CubicalComplex & codomain, 
                 const RectMap & F );

  // When used for filling, build only the quotient needed by the solver.
  FiberComplex ( const boost::unordered_set < Index > & X_nbs, 
                 const boost::unordered_set < Index > & A_nbs,
                 const CubicalComplex & domain,
                 CubicalComplex & codomain, 
                 const std::unordered_map < uint64_t, std::vector < uint64_t > > & F,
                 bool for_filling = false );

  virtual ~FiberComplex ( void ) {}; 
  
  // Inclusion and Projection
  void include ( Chain * output, const Chain & input ) const;
  void project ( Chain * output, const Chain & input ) const;
  Index choose ( void ) const { return choose_; }
  // Boundary and Coboundary
  virtual void 
  boundary ( Chain * output, const Index input, int dim ) const;
  virtual void 
  coboundary ( Chain * output, const Index input, int dim ) const;
  
  virtual void 
  boundary ( Chain * output, const Chain & input ) const;  
  
  virtual void 
  coboundary ( Chain * output, const Chain & input ) const;
  
  bool acyclic ( void ) /* const */;
  bool trivial ( void ) /* const */;
  bool acyclic_or_trivial ( void ) /* const */;
  static bool acyclic_or_trivial ( MorseComplex& reduction );
  
  Chain preboundary ( const Chain & input ) /* const */;
  Chain preboundary ( const Chain& input, MorseComplex& reduction );
  
  //convenience
  Chain include ( const Chain & input ) const {
    Chain output; include ( &output, input ); return output;
  }
  Chain project ( const Chain & input ) const {
    Chain output; project ( &output, input ); return output;
  }  
  
  Rect geometry ( Index i, int dim ) const;

};
/********************
 *   DEFINITIONS    *
 ********************/

template < class RectMap >
inline FiberComplex::FiberComplex
( const boost::unordered_set < Index > & X_nbs, 
  const boost::unordered_set < Index > & A_nbs,
  const CubicalComplex & domain,
  CubicalComplex & codomain, 
  const RectMap & F ) {
  // Construct the fiber
  supercomplex_ = & codomain;
  int D = supercomplex_ -> dimension ();
  
  // Determine Images
  
  std::vector < boost::unordered_set < Index > > X_image ( D + 1 );
  std::vector < boost::unordered_set < Index > > A_image ( D + 1 );
  BOOST_FOREACH ( Index i, X_nbs ) {
    std::insert_iterator<boost::unordered_set < Index > > 
    ii ( X_image [ D ], X_image [ D ] . end () );
    codomain . cover ( ii, F ( domain . geometry ( i, D ) ) );
  }
  BOOST_FOREACH ( Index i, A_nbs ) {
    std::insert_iterator<boost::unordered_set < Index > > 
    ii ( A_image [ D ], A_image [ D ] . end () );
    codomain . cover ( ii, F ( domain . geometry ( i, D ) ) );
  }

  // Find the combinatorial closures of X and A
  closure ( X_image, * supercomplex_ );
  closure ( A_image, * supercomplex_ );
  
  // Pick choose   (dirty: this makes sense only if A is empty )
#if 0
  // DEBUG //////////////////////////////////////
  if ( X_image [ 0 ]. size () == 0 ) {
    // TAKE OUT
    return;
    std::cout << "FiberComplex failure. The fiber doesn't even have a single vertex.\n";
    BOOST_FOREACH ( Index i, X_nbs ) {
      std::cout << domain . geometry ( i, D ) << " --> " << F ( domain . geometry ( i, D ) ) << "\n";
    }
    std::cout << "Size of codomain: " << codomain . size () << "\n";
    for ( int d = 0; d <= D; ++ d ) {
      std::cout << " size of X_image [ " << d << " ] = " << X_image [ d ] . size () << "\n";
    }
    for ( int d = 0; d <= D; ++ d ) {
      std::cout << " size of A_image [ " << d << " ] = " << A_image [ d ] . size () << "\n";
    }
#if 0
    {
      ComplexVisualization * cv = new ComplexVisualization ( "Codomain Picture." );
      std::cout << "About to draw complex.\n";
      cv -> drawComplex ( codomain, 100 );
      BOOST_FOREACH ( Index i, X_nbs ) {
        cv -> drawRect ( domain . geometry ( i, D ), 200 );
        //cv -> drawRect ( F ( domain . geometry ( i, D ) ) , 200 );
      }
      // explore a minute here
      cv -> explore ();
      delete cv;
    }
#endif
    return;
    exit ( 1 );
  }
  ///////////////////////////////////////////////
#endif
  
  //std::cout << "PICKING CHOOSE\n";
  if ( X_image [ 0 ] . size () > 0 ) {
    choose_ = * X_image [ 0 ] . begin ();
  }
  //std::cout << "CHOOSE PICKED\n";
  // Initialize bitmap.
  for ( int d = 0; d <= D; ++ d ) {
    boost::unordered_set<Index> relative_cells;
    for (Index i : X_image[d]) relative_cells.insert(i);
    for (Index i : A_image[d]) relative_cells.erase(i);
    for (Index i : relative_cells) insertCell(i, d);
  }
  
}

inline FiberComplex::FiberComplex
( const boost::unordered_set < Index > & X_nbs,
  const boost::unordered_set < Index > & A_nbs,
  const CubicalComplex & domain,
  CubicalComplex & codomain,
  const std::unordered_map < uint64_t, std::vector < uint64_t > > & F,
  bool for_filling ) {
  // Construct the fiber
  supercomplex_ = & codomain;
  int D = supercomplex_ -> dimension ();
  if (for_filling && !A_nbs.empty() && !codomain.hasPeriodicAxis()) {
    buildForFilling(X_nbs, A_nbs, F);
    return;
  }

  // Determine Images

  std::vector < boost::unordered_set < Index > > X_image ( D + 1 );
  std::vector < boost::unordered_set < Index > > A_image ( D + 1 );
  BOOST_FOREACH ( Index i, X_nbs ) {
    // std::insert_iterator<boost::unordered_set < Index > > ii ( X_image [ D ], X_image [ D ] . end () );
    for ( const auto& j : F . at ( i ) ) {
      X_image [ D ] . insert ( j );
      // *ii = j;
    }
  }
  BOOST_FOREACH ( Index i, A_nbs ) {
    // std::insert_iterator<boost::unordered_set < Index > > ii ( A_image [ D ], A_image [ D ] . end () );
    for ( const auto& j : F . at ( i ) ) {
      A_image [ D ] . insert ( j );
      // *ii = j;
    }
  }

  // std::cout << "Size of codomain: " << codomain . size () << std::endl;
  // for ( int d = 0; d <= D; ++ d ) {
  //   std::cout << " size of X_image [ " << d << " ] = " << X_image [ d ] . size () << std::endl;
  // }
  // for ( int d = 0; d <= D; ++ d ) {
  //   std::cout << " size of A_image [ " << d << " ] = " << A_image [ d ] . size () << std::endl;
  // }

  // Find the combinatorial closures of X and A
  closure ( X_image, * supercomplex_ );
  closure ( A_image, * supercomplex_ );

  // std::cout << "Size of codomain: " << codomain . size () << std::endl;
  // for ( int d = 0; d <= D; ++ d ) {
  //   std::cout << " size of X_image [ " << d << " ] = " << X_image [ d ] . size () << std::endl;
  // }
  // for ( int d = 0; d <= D; ++ d ) {
  //   std::cout << " size of A_image [ " << d << " ] = " << A_image [ d ] . size () << std::endl;
  // }

  // std::cout << "PICKING CHOOSE\n";
  if ( X_image [ 0 ] . size () > 0 ) {
    choose_ = * X_image [ 0 ] . begin ();
  }
  // std::cout << "CHOOSE PICKED\n";
  // Initialize bitmap
  for ( int d = 0; d <= D; ++ d ) {
    boost::unordered_set<Index> relative_cells;
    for (Index i : X_image[d]) relative_cells.insert(i);
    for (Index i : A_image[d]) relative_cells.erase(i);
    for (Index i : relative_cells) insertCell(i, d);
  }
}

inline void FiberComplex::buildForFilling (
    const boost::unordered_set<Index>& X_nbs, const boost::unordered_set<Index>& A_nbs,
    const std::unordered_map<Index, std::vector<Index>>& images ) {
  const auto& target = *supercomplex_;
  const int D = target.dimension();
  // cl(X) minus cl(A) equals cl(X minus A) minus cl(A). Only faces
  // incident to a surviving top cube can contribute to the quotient.
  boost::unordered_set<uint64_t> exit_cubes;
  for (Index i : A_nbs)
    for (Index j : images.at(i)) exit_cubes.insert(target.indexToCell(j, D));
  std::vector<boost::unordered_set<Index>> surviving(D + 1);
  for (Index i : X_nbs)
    for (Index j : images.at(i))
      if (!exit_cubes.count(target.indexToCell(j, D))) surviving[D].insert(j);
  if (surviving[D].empty()) return;
  closure(surviving, target);

  std::vector<uint64_t> step(D), origin(D, 0);
  const auto origin_address = target.cubeToAddress(origin);
  for (int d = 0; d < D; ++d) {
    origin[d] = 1;
    step[d] = target.cubeToAddress(origin) - origin_address;
    origin[d] = 0;
  }
  const auto top_axes = (uint64_t{1} << D) - 1;
  auto in_exit = [&](uint64_t address) {
    const auto fixed_axes = top_axes & ~address;
    // The nonperiodic lower buffer cannot contain an exit cube.
    for (auto subset = fixed_axes; ; subset = (subset - 1) & fixed_axes) {
      auto candidate = address | top_axes;
      for (int d = 0; d < D; ++d)
        if (subset & (uint64_t{1} << d)) candidate -= step[d];
      if (exit_cubes.count(candidate)) return true;
      if (subset == 0) return false;
    }
  };
  for (int d = 0; d <= D; ++d)
    for (Index cell : surviving[d])
      if (!in_exit(target.indexToCell(cell, d))) insertCell(cell, d);
}

inline void FiberComplex::include ( Chain * output, 
                                 const Chain & input ) const {
  int dim = input . dimension ();
  BOOST_FOREACH ( Term term, input () ) {
    term . index () = indexToCell ( term . index (), dim );
    (*output) += term;
  }
  output -> dimension () = dim;
}

inline void FiberComplex::project ( Chain * output, 
                                 const Chain & input ) const {
  int dim = input . dimension ();
  BOOST_FOREACH ( Term term, input () ) {
    const auto index = cellToIndex(term.index(), dim);
    if (index < size(dim)) (*output) += Term(index, term.coef());
  }
  output -> dimension () = dim;
}
  
  
  
  inline bool FiberComplex::trivial ( void ) /* const */ {
    Generators_t gen = MorseGenerators ( *this );
    //std::cout << "acyclic:\n";
    //std::cout << gen . size () << " and " << dimension () << "\n";
    assert ( (int) gen . size () == dimension () + 1 );
    for ( int d = 0; d <= dimension (); ++ d ) {
      if ( gen [ d ] . size () > 0 ) return false;
    }
    return true;
  }
  
  inline bool FiberComplex::acyclic ( void ) /* const */ {
     //std::cout << "Fiber dimension = " << dimension () << " Fiber size = " << size () << "\n";

    Generators_t gen = MorseGenerators ( *this );
    //std::cout << "acyclic:\n";
    //std::cout << gen . size () << " and " << dimension () << "\n";
    assert ( (int) gen . size () == dimension () + 1 );
    bool result = true;
    if ( gen . size () == 0 ) result = false;
    if ( gen [ 0 ] . size () != 1 ) result = false;
    for ( int d = 1; d <= dimension (); ++ d ) {
      if ( gen [ d ] . size () > 0 ) result = false;
    }
    // DEBUG BEGIN
    /*
    if ( not result ) {
      std::cout << "FiberComplex::acyclic. gen.size=" << 
        gen . size () << " and dim=" << dimension () << "\n";
      for ( int d = 0; d <= dimension (); ++ d ) {
        std::cout << gen [ d ] . size () << " ";
      }
      std::cout << "\n";
      std::cout << "Fiber size = " << size () << "\n";
    }
    */
    // DEBUG END
    return result;
  }
  
  inline bool FiberComplex::acyclic_or_trivial ( void ) /* const */ {
    MorseComplex reduction(*this);
    return acyclic_or_trivial(reduction);
  }

  inline bool FiberComplex::acyclic_or_trivial ( MorseComplex& reduction ) {
    const auto groups = SmithGenerators(reduction, reduction.dimension());
    return groups.empty() || (groups[0].size() <= 1 &&
        std::all_of(groups.begin() + 1, groups.end(), [](const auto& degree) { return degree.empty(); }));
  }
  
inline Chain FiberComplex::preboundary
( const Chain & input ) /* const */ {
  //std::cout << "FiberComplex preboundary, dim = " << input . dimension () <<"\n";
  MorseComplex reduction ( *this );
  return preboundary(input, reduction);
}

inline Chain FiberComplex::preboundary ( const Chain& input, MorseComplex& reduction ) {
  Chain canon;
  Chain gamma;
  reduction . flow ( &canon, &gamma, input );
  //std::cout << "canon/gamma dim = " << canon . dimension () << ", " << gamma . dimension () << "\n";
  Chain lowered = reduction . project ( canon );
  //std::cout << "lowered.dim = " << lowered . dimension () << "\n";
  Chain morse_preboundary = SmithPreboundary ( lowered, reduction );
  //std::cout << "smithpreboundary.dim = " << morse_preboundary . dimension () << "\n";
  Chain lifted = reduction . lift ( morse_preboundary );
  //std::cout << "lifted.dim = " << lifted . dimension () << "\n";
  Chain result = lifted - gamma;
  Chain discrepancy = simplify ( input - boundary ( result ) );
  if ( not discrepancy () . empty () ) {
    // The Morse-reduced route can fail on rare fibers. Fall back to the
    // direct Smith-normal-form solve on the full fiber complex: slower
    // (it factors the full boundary matrix) but it solves the defining
    // equation boundary(result) == input directly. Only if that too fails
    // is the input genuinely not a boundary, which is a real error.
    result = SmithPreboundary ( input, *this );
    discrepancy = simplify ( input - boundary ( result ) );
    if ( not discrepancy () . empty () ) {
      throw std::runtime_error (
        "FiberComplex::preboundary failed boundary validation on both the "
        "Morse-reduced and the full-complex Smith routes" );
    }
  }
  return result;
}

inline void FiberComplex::boundary 
( Chain * output, const Index input, int dim ) const {
  project(output, supercomplex_->boundary(indexToCell(input, dim), dim));
}

inline void FiberComplex::coboundary 
( Chain * output, const Index input, int dim ) const {
  project(output, supercomplex_->coboundary(indexToCell(input, dim), dim));
}

inline void FiberComplex::boundary 
( Chain * output, const Chain & input ) const {
  project ( output, supercomplex_ -> boundary ( include ( input ) ) );
}

inline void FiberComplex::coboundary 
( Chain * output, const Chain & input ) const {
  project ( output, supercomplex_ -> coboundary ( include ( input ) ) );
  
}

inline Rect FiberComplex::geometry ( Index i, int dim ) const {
  //std::cout << "FiberComplex::geometry ( " << i << ", " << dim << ")\n";
  return supercomplex_ -> geometry ( indexToCell ( i, dim ), dim );
}

} // namespace chomp

#endif
