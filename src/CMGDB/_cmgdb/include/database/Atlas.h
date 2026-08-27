#ifndef ATLAS_H
#define ATLAS_H

#include <iostream>
#include <cstdint>

#include <algorithm>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <vector>
#include <unordered_map>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include "Grid.h"
#include "TreeGrid.h"
#include "PointerGrid.h"
#include "RankSelect.h"
#include "Geo.h"
#include "AtlasGeo.h"

/** One equal-depth dyadic cell in a chart rectangle.
 *
 * `axis_depth` is the number of bisections in every coordinate and
 * `coordinates[d]` is the dyadic index in coordinate `d`.  The corresponding
 * interval is
 *
 *   [lower[d] + coordinates[d] * width[d] / 2^axis_depth,
 *    lower[d] + (coordinates[d] + 1) * width[d] / 2^axis_depth].
 *
 * Different cells in one active family may have different axis depths, but
 * they must be an antichain: duplicate cells are ignored and a cell may not
 * contain another selected cell.
 */
struct AtlasDyadicCell {
  uint64_t axis_depth;
  std::vector<uint64_t> coordinates;

  AtlasDyadicCell ( void ) : axis_depth ( 0 ) {}
  AtlasDyadicCell ( uint64_t depth, std::vector<uint64_t> indices )
    : axis_depth ( depth ), coordinates ( std::move ( indices ) ) {}
};

/// class Atlas
///   Grid data structure which stores other Grids as "charts"
///   Charts are added with "add_chart_" method.
///   Each added chart is assigned a "chart_id" by the user which may be any integer.
///   The integers must be unique to the chart, but need not be contiguous.
///   Once the charts have been added the "finalize" method must be called
///     to make the data structure usable.
///   Alternatively, charts may be loaded from a file with the "import_charts" method. 

class Atlas : public Grid { 

public:
	typedef uint64_t GridElement;
  typedef boost::counting_iterator < GridElement > iterator;
  typedef iterator const_iterator;
  typedef uint64_t size_type;
  typedef std::shared_ptr<TreeGrid> Chart;

  // Contructor/ Desctructor
  Atlas ( void ) { }
  ~Atlas ( void ) { }

  // Methods inherited from Grid
  virtual Atlas * clone ( void ) const;
  virtual void subdivide ( void );
  virtual Grid * subgrid ( const std::deque < GridElement > & grid_elements ) const;
  virtual std::vector<GridElement> subset ( const Grid & other ) const;
  virtual std::shared_ptr<Geo> geometry ( GridElement ge ) const;  
  virtual std::vector<Grid::GridElement> cover ( const Geo & geo ) const;
  using Grid::geometry;
  using Grid::cover;

  // Atlas-Specific functionality

  /// Typedefs for Atlas
  typedef std::unordered_map<size_type, Chart>::const_iterator ChartIterator;
  typedef boost::iterator_range<ChartIterator> ChartIteratorRange;
  typedef std::pair <size_type, Chart > IdChartPair;

  /// chart
  ///   Accessor method for chart via chart_id
  Chart & 
  chart ( size_type chart_id );  

  /// chart (const method)
  ///   Accessor method for chart via chart_id
  Chart const& 
  chart ( size_type chart_id ) const;

  /// clear
  ///   Revert to an empty Atlas structure
  void 
  clear ( void );

  /// importCharts
  ///   Read an input file detailing an Atlas structure 
  ///   and create the appropriate data structure.
  void 
  importCharts ( const char * inputfile );
  
  /// add_chart
  ///   add a chart to the Atlas data structure
  void 
  add_chart ( size_type id, const RectGeo & rect);

  /// add_chart with per-coordinate periodicity
  void
  add_chart ( size_type id, const RectGeo & rect,
              const std::vector<bool> & periodic );

  /// add_chart
  ///   add a chart to the Atlas data structure
  void 
  add_chart ( size_type id, int dimension, const RectGeo & rect);

  /// Replace one chart's active grid by a selected dyadic antichain.
  ///
  /// This constructs the compressed prefix tree directly; it does not first
  /// allocate the complete rectangular grid at the requested depth.  An empty
  /// vector makes the chart empty while retaining its chart id and bounds.
  void
  set_chart_active_dyadic_cells (
      size_type id, const std::vector<AtlasDyadicCell> & cells );

  /// listCharts
  ///   Print chart information to std::cout
  void 
  list_charts ( void ) const;

  /// numCharts
  ///   return number of charts
  uint64_t 
  numCharts ( void ) const;

  /// finalize
  ///    Finalize indexing to make Atlas data structure usable.
  void 
  finalize ( void );

  /// memory
  ///   Return memory usage
  virtual uint64_t 
  memory ( void ) const;

  /// charts
  ///   Return an iterator range which iterates through pairs (chart_id, chart)
  ChartIteratorRange
  charts ( void ) const;

private:
  // chart information
  std::unordered_map < size_type, Chart > charts_; 
  // indexing information
  std::unordered_map<size_type, size_type> chart_id_to_index_;
  std::vector<size_type> chart_index_to_id_;
  RankSelect convert_;
  // indexing methods
  GridElement 
  Chart_to_Atlas_GridElement_ ( GridElement const& chart_ge, 
                                size_type const& chart_id ) const;

  std::pair < size_type, GridElement > 
  Atlas_to_Chart_GridElement_ ( GridElement const& atlas_ge ) const;

};

namespace cmgdb_atlas_active_detail {

struct TrieNode {
  bool terminal = false;
  std::unique_ptr<TrieNode> child [ 2 ];
};

inline bool
is_prefix ( const std::vector<bool> & prefix,
            const std::vector<bool> & value ) {
  return prefix . size () <= value . size () and
    std::equal ( prefix . begin (), prefix . end (), value . begin () );
}

inline void
emit_compressed_tree ( const TrieNode & node, CompressedTree * output ) {
  if ( node . terminal ) {
    output -> leaf_sequence . push_back ( false );
    output -> valid_sequence . push_back ( true );
    return;
  }
  output -> leaf_sequence . push_back ( true );
  for ( int branch = 0; branch < 2; ++ branch ) {
    if ( node . child [ branch ] ) {
      emit_compressed_tree ( * node . child [ branch ], output );
    } else {
      output -> leaf_sequence . push_back ( false );
      output -> valid_sequence . push_back ( false );
    }
  }
}

inline std::vector<bool>
dyadic_path ( const AtlasDyadicCell & cell, size_t dimension ) {
  if ( cell . coordinates . size () != dimension ) {
    throw std::invalid_argument (
      "Atlas dyadic cell needs one coordinate index per chart dimension" );
  }
  // A uint64_t coordinate can safely be range-checked against 2^depth only
  // through depth 63.  This is already far beyond any realizable grid.
  if ( cell . axis_depth > 63 ) {
    throw std::invalid_argument (
      "Atlas dyadic cell axis_depth must be at most 63" );
  }
  if ( dimension != 0 and
       cell . axis_depth > std::numeric_limits<size_t>::max () / dimension ) {
    throw std::invalid_argument ( "Atlas dyadic cell path is too deep" );
  }
  const uint64_t cells_per_axis = uint64_t ( 1 ) << cell . axis_depth;
  for ( size_t d = 0; d < dimension; ++ d ) {
    if ( cell . coordinates [ d ] >= cells_per_axis ) {
      std::ostringstream message;
      message << "Atlas dyadic coordinate " << cell . coordinates [ d ]
              << " is outside [0," << cells_per_axis << ") at axis_depth "
              << cell . axis_depth;
      throw std::invalid_argument ( message . str () );
    }
  }

  std::vector<bool> path;
  path . reserve ( dimension * cell . axis_depth );
  for ( uint64_t level = 0; level < cell . axis_depth; ++ level ) {
    const uint64_t shift = cell . axis_depth - level - 1;
    for ( size_t d = 0; d < dimension; ++ d ) {
      path . push_back ( ( cell . coordinates [ d ] >> shift ) & 1U );
    }
  }
  return path;
}

inline std::shared_ptr<CompressedTreeGrid>
make_active_chart (
    const RectGeo & bounds,
    const std::vector<bool> & periodic,
    const std::vector<AtlasDyadicCell> & cells ) {
  std::shared_ptr<CompressedTreeGrid> result ( new CompressedTreeGrid );
  result -> bounds () = bounds;
  result -> periodicity () = periodic;

  std::vector<std::vector<bool>> paths;
  paths . reserve ( cells . size () );
  for ( const AtlasDyadicCell & cell : cells ) {
    paths . push_back ( dyadic_path ( cell, bounds . dimension () ) );
  }
  std::sort ( paths . begin (), paths . end () );
  paths . erase ( std::unique ( paths . begin (), paths . end () ), paths . end () );
  for ( size_t i = 1; i < paths . size (); ++ i ) {
    if ( is_prefix ( paths [ i - 1 ], paths [ i ] ) ) {
      throw std::invalid_argument (
        "Atlas active dyadic cells must be an antichain (no selected cell may contain another)" );
    }
  }

  if ( paths . empty () ) {
    result -> tree () -> leaf_sequence . push_back ( false );
    result -> tree () -> valid_sequence . push_back ( false );
    return result;
  }

  TrieNode root;
  for ( const std::vector<bool> & path : paths ) {
    TrieNode * node = & root;
    for ( bool bit : path ) {
      const int branch = bit ? 1 : 0;
      if ( not node -> child [ branch ] ) {
        node -> child [ branch ] . reset ( new TrieNode );
      }
      node = node -> child [ branch ] . get ();
    }
    node -> terminal = true;
  }
  emit_compressed_tree ( root, result -> tree () . get () );
  return result;
}

} // namespace cmgdb_atlas_active_detail

inline Atlas * 
Atlas::clone ( void ) const {
  Atlas * newAtlas = new Atlas;
  for ( IdChartPair const& pair : charts () ) {
    std::shared_ptr<TreeGrid> chart_ptr ( (TreeGrid *) (pair . second -> clone ()) );
    newAtlas -> chart ( pair . first ) = chart_ptr;    
  }
  newAtlas -> finalize ();
  return newAtlas;
}


inline void 
Atlas::subdivide ( void ) { 
  for ( IdChartPair const& pair : charts () ) {
    pair . second -> subdivide ( );  
  }  
  finalize ();
}

inline Grid * 
Atlas::subgrid ( const std::deque < GridElement > & grid_elements ) const {
  std::unordered_map < size_type, std::deque < GridElement > > chart_grid_elements;
  for ( GridElement ge : grid_elements ) {
    std::pair < size_type, GridElement > atlas_ge = Atlas_to_Chart_GridElement_ ( ge );
    chart_grid_elements [ atlas_ge . first ] . push_back ( atlas_ge . second );
  }
  Atlas * newAtlas = new Atlas;
  for ( IdChartPair const& pair : charts () ) {
    Grid * subchart = pair . second -> subgrid ( chart_grid_elements [ pair . first ] );
    newAtlas -> charts_ [ pair . first ] = 
      std::shared_ptr<TreeGrid> ( (TreeGrid *) subchart );
  }  
  newAtlas -> finalize ();
  return (Grid *) newAtlas;
}

inline std::vector<Grid::GridElement> 
Atlas::subset ( const Grid & other ) const {
  const Atlas & otherAtlas = dynamic_cast<const Atlas &> (other);
  std::vector<Grid::GridElement> result;
  for ( IdChartPair const& pair : charts () ) {
    std::vector<Grid::GridElement> chart_subset = 
      pair . second -> subset ( * otherAtlas . charts_ . find ( pair . first ) -> second );
    for ( Grid::GridElement ge : chart_subset ) {
      result . push_back ( Chart_to_Atlas_GridElement_ ( ge, pair . first ) );
    }
  }
  return result;
}

inline std::shared_ptr<Geo> 
Atlas::geometry ( Grid::GridElement ge ) const {
  std::pair < size_type, GridElement > chartge;
  chartge = Atlas_to_Chart_GridElement_ ( ge );
  RectGeo rect = * std::dynamic_pointer_cast < RectGeo > 
    ( charts_ . find ( chartge . first ) -> second -> geometry ( chartge . second ) );
  return std::shared_ptr<Geo> ( new AtlasGeo ( chartge.first, rect ) );
}

inline std::vector<Grid::GridElement>
Atlas::cover ( const Geo & geo ) const { 
  const AtlasGeo & atlas_geo = dynamic_cast<const AtlasGeo &> ( geo );
  std::vector<Grid::GridElement> result;
  size_type chart_id_of_geo = atlas_geo . id ();
  const Chart & chart_of_geo = charts_ . find ( chart_id_of_geo ) -> second;
  if ( chart_of_geo -> size () == 0 ) return result;
  std::vector < GridElement > listge = chart_of_geo -> cover ( atlas_geo . rect() );
  for ( Grid::GridElement chart_ge : listge ) {
    GridElement newge = Chart_to_Atlas_GridElement_ ( chart_ge , atlas_geo . id() );
    result . push_back ( newge );
  }
  return result;
}

inline void 
Atlas::add_chart ( size_type id, const RectGeo & rect ) {
  charts_ [ id ] = std::shared_ptr<TreeGrid> ( new PointerGrid );
  charts_ [ id ] -> initialize ( rect );
}

inline void
Atlas::add_chart ( size_type id, const RectGeo & rect,
                   const std::vector<bool> & periodic ) {
  charts_ [ id ] = std::shared_ptr<TreeGrid> ( new PointerGrid );
  charts_ [ id ] -> initialize ( rect, periodic );
}

inline void 
Atlas::add_chart ( size_type id, int dimension, const RectGeo & rect ) {
  charts_ [ id ] = std::shared_ptr<TreeGrid> ( new PointerGrid );
  charts_ [ id ] -> initialize ( rect );
  charts_ [ id ] -> dimension  ( ) = dimension;
}

inline void
Atlas::set_chart_active_dyadic_cells (
    size_type id, const std::vector<AtlasDyadicCell> & cells ) {
  auto chart_it = charts_ . find ( id );
  if ( chart_it == charts_ . end () ) {
    std::ostringstream message;
    message << "Atlas has no chart " << id;
    throw std::invalid_argument ( message . str () );
  }
  const RectGeo bounds = chart_it -> second -> bounds ();
  const std::vector<bool> periodic = chart_it -> second -> periodicity ();
  std::shared_ptr<CompressedTreeGrid> compressed =
    cmgdb_atlas_active_detail::make_active_chart ( bounds, periodic, cells );
  std::shared_ptr<PointerGrid> active ( new PointerGrid );
  active -> assign ( compressed );
  chart_it -> second = active;
  finalize ();
}

inline void 
Atlas::list_charts ( void ) const {
  std::cout << "\nList of charts :\n";
  for ( IdChartPair const& pair : charts () ) {
    std::cout << "index = " << pair . first << " , ";
    std::cout << "bounds = " << pair . second -> bounds ( ) << " , "; 
    std::cout << "number of GridElements = " << pair . second -> size ( ) << "\n";
  }  
}

inline uint64_t 
Atlas::numCharts ( void ) const {
  return charts_ . size ();
}

inline void 
Atlas::importCharts ( const char * inputfile ) {
  using boost::property_tree::ptree;
  ptree pt;
  std::ifstream input ( inputfile );
  read_xml(input, pt);

  unsigned int dimension = pt.get<int>("atlas.dimension");
  std::cout << "Dimension : " << dimension << "\n";

  std::vector < double > lower_bounds, upper_bounds;
  lower_bounds . resize ( dimension );
  upper_bounds . resize ( dimension );

  for ( ptree::value_type & v : pt.get_child("atlas.listcharts") ) {
    // extract the strings
    std::string idstr = v . second . get_child ( "id" ) . data ( );
    std::string lbstr = v . second . get_child ( "lbounds" ) . data ( );
    std::string ubstr = v . second . get_child ( "ubounds" ) . data ( );
    std::stringstream idss ( idstr );
    std::stringstream lbss ( lbstr );
    std::stringstream ubss ( ubstr );
    size_type id;
    idss >> id;
    for ( unsigned int d = 0; d < dimension; ++ d ) {
      lbss >> lower_bounds [ d ];
      ubss >> upper_bounds [ d ];
    }
    // add the new chart 
    add_chart ( id, RectGeo(dimension,lower_bounds,upper_bounds) ); 
  }
  finalize ();
}

inline Atlas::ChartIteratorRange
Atlas::charts ( void ) const {
  return boost::make_iterator_range ( charts_ . begin (), charts_ . end () );
}

inline Atlas::Chart & 
Atlas::chart ( size_type chart_id ) {
  return charts_ [ chart_id ];
}

inline const Atlas::Chart & 
Atlas::chart ( size_type chart_id ) const {
  return charts_ . find ( chart_id ) -> second;
}

inline void 
Atlas::clear ( void ) {
  charts_ . clear ();
  finalize ();
}

inline uint64_t 
Atlas::memory ( void ) const {
  uint64_t result = 0;
  for ( IdChartPair const& chartpair : charts () ) {
    result += chartpair . second -> memory ();
  }
  return result;
}

inline void 
Atlas::finalize ( void ) { 
  chart_id_to_index_ . clear ();
  chart_index_to_id_ . clear ();
  size_ = 0;
  std::vector<size_type> nonempty_chart_ids;
  for ( IdChartPair const& pair : charts () ) {
    if ( pair . second -> size () != 0 ) {
      nonempty_chart_ids . push_back ( pair . first );
    }
  }
  // Stable cell numbering matters to downstream adjacency data and plots.
  // unordered_map iteration order is not a reproducible chart order.
  std::sort ( nonempty_chart_ids . begin (), nonempty_chart_ids . end () );
  for ( size_type chart_index = 0;
        chart_index < nonempty_chart_ids . size ();
        ++ chart_index ) {
    const size_type chart_id = nonempty_chart_ids [ chart_index ];
    const size_type chart_size = charts_ [ chart_id ] -> size ();
    size_ += chart_size;
    chart_id_to_index_ [ chart_id ] = chart_index;
    chart_index_to_id_ . push_back ( chart_id );
  }
  std::vector<bool> bits ( size_ );
  size_type s = 0;
  for ( size_type chart_index = 0; chart_index < chart_index_to_id_ . size (); ++ chart_index ) {
    bits [ s ] = 1;
    s += charts_ [ chart_index_to_id_ [ chart_index ] ] -> size ();
  }
  convert_ . assign ( bits );
}

inline Atlas::GridElement 
Atlas::Chart_to_Atlas_GridElement_ ( GridElement const& chart_ge, 
                                     size_type const& chart_id ) const {
  size_type chart_index = chart_id_to_index_ . find ( chart_id ) -> second;
  return convert_ . select ( chart_index ) + chart_ge;
}

inline std::pair < Atlas::size_type, Atlas::GridElement > 
Atlas::Atlas_to_Chart_GridElement_ ( GridElement const& atlas_ge ) const {
  Atlas::size_type chart_index = convert_ . rank ( atlas_ge + 1 ) - 1;
  Atlas::GridElement chart_ge = atlas_ge - convert_ . select ( chart_index );
  return std::make_pair ( chart_index_to_id_[chart_index], chart_ge );
}

#endif
