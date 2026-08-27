#ifndef CMDB_ATLAS_MODEL_H
#define CMDB_ATLAS_MODEL_H

#include "Atlas.h"
#include "AtlasModelMapF.h"
#include "Map.h"
#include "RectGeo.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

/** A CMGDB model whose phase space is a finite disjoint union of charts.
 *
 * AtlasModel deliberately has a separate API from Model: the ordinary
 * Euclidean rectangle-map API remains unchanged.  Charts must be installed
 * before the tagged finite-union map is set.
 */
struct TaggedDyadicCell {
  uint64_t chart_id;
  uint64_t axis_depth;
  std::vector<uint64_t> coordinates;

  TaggedDyadicCell ( void ) : chart_id ( 0 ), axis_depth ( 0 ) {}
  TaggedDyadicCell ( uint64_t chart,
                     uint64_t depth,
                     std::vector<uint64_t> indices )
    : chart_id ( chart ),
      axis_depth ( depth ),
      coordinates ( std::move ( indices ) ) {}
};

class AtlasModel {
public:
  typedef AtlasModelMapF::Callback Callback;

  explicit AtlasModel ( int phase_subdiv )
    : AtlasModel ( phase_subdiv, phase_subdiv, phase_subdiv, 10000 ) {}

  AtlasModel ( int phase_subdiv_min,
               int phase_subdiv_max,
               int phase_subdiv_init,
               int phase_subdiv_limit )
    : phase_subdiv_min_ ( phase_subdiv_min ),
      phase_subdiv_max_ ( phase_subdiv_max ),
      phase_subdiv_init_ ( phase_subdiv_init ),
      phase_subdiv_limit_ ( phase_subdiv_limit ),
      phase_space_ ( new Atlas ) {
    if ( phase_subdiv_init < 0 or phase_subdiv_min < 0 or
         phase_subdiv_max < 0 ) {
      throw std::invalid_argument ( "AtlasModel subdivision depths must be nonnegative" );
    }
    if ( phase_subdiv_init > phase_subdiv_min or
         phase_subdiv_min > phase_subdiv_max ) {
      throw std::invalid_argument (
        "AtlasModel requires phase_subdiv_init <= phase_subdiv_min <= phase_subdiv_max" );
    }
    if ( phase_subdiv_limit <= 0 ) {
      throw std::invalid_argument ( "AtlasModel phase_subdiv_limit must be positive" );
    }
  }

  void add_chart ( uint64_t chart_id,
                   const std::vector<double> & lower_bounds,
                   const std::vector<double> & upper_bounds,
                   const std::vector<bool> & periodic ) {
    if ( map_ or active_subgrid_configured_ ) {
      throw std::logic_error (
        "AtlasModel charts cannot be changed after set_map or set_active_subgrid; construct a new model" );
    }
    if ( chart_dimensions_ . count ( chart_id ) != 0 ) {
      std::ostringstream message;
      message << "AtlasModel chart " << chart_id << " already exists";
      throw std::invalid_argument ( message . str () );
    }
    if ( lower_bounds . empty () or
         lower_bounds . size () != upper_bounds . size () ) {
      throw std::invalid_argument (
        "AtlasModel chart bounds must be nonempty vectors of equal length" );
    }
    if ( periodic . size () != lower_bounds . size () ) {
      throw std::invalid_argument (
        "AtlasModel chart periodicity must have one flag per coordinate" );
    }
    for ( size_t d = 0; d < lower_bounds . size (); ++ d ) {
      if ( not std::isfinite ( lower_bounds [ d ] ) or
           not std::isfinite ( upper_bounds [ d ] ) ) {
        throw std::invalid_argument ( "AtlasModel chart bounds must be finite" );
      }
      if ( lower_bounds [ d ] >= upper_bounds [ d ] ) {
        std::ostringstream message;
        message << "AtlasModel chart " << chart_id
                << " requires lower < upper in coordinate " << d;
        throw std::invalid_argument ( message . str () );
      }
    }

    RectGeo rectangle (
      lower_bounds . size (), lower_bounds, upper_bounds );
    phase_space_ -> add_chart ( chart_id, rectangle, periodic );
    phase_space_ -> finalize ();
    chart_dimensions_ [ chart_id ] = lower_bounds . size ();
    chart_ids_ . push_back ( chart_id );
    std::sort ( chart_ids_ . begin (), chart_ids_ . end () );
  }

  void add_chart ( uint64_t chart_id,
                   const std::vector<double> & lower_bounds,
                   const std::vector<double> & upper_bounds ) {
    add_chart ( chart_id, lower_bounds, upper_bounds,
                std::vector<bool> ( lower_bounds . size (), false ) );
  }

  void set_map ( Callback callback ) {
    if ( chart_dimensions_ . empty () ) {
      throw std::logic_error (
        "AtlasModel.set_map requires at least one chart" );
    }
    if ( not callback ) {
      throw std::invalid_argument ( "AtlasModel.set_map requires a callback" );
    }
    map_ . reset ( new AtlasModelMapF ( chart_dimensions_, std::move ( callback ) ) );
  }

  /** Replace the default full chart roots by an explicitly selected family of
   * dyadic chart cells.
   *
   * Every registered chart remains present.  A chart omitted from `cells`
   * becomes an empty chart.  The family in each chart must be an antichain.
   * Construction is transactional and uses a compressed prefix tree directly;
   * no complete rectangular grid at the maximum requested depth is allocated.
   */
  void set_active_subgrid ( const std::vector<TaggedDyadicCell> & cells ) {
    if ( map_ ) {
      throw std::logic_error (
        "AtlasModel active subgrid cannot be changed after set_map" );
    }
    if ( chart_dimensions_ . empty () ) {
      throw std::logic_error (
        "AtlasModel.set_active_subgrid requires at least one chart" );
    }

    std::unordered_map<uint64_t, std::vector<AtlasDyadicCell>> by_chart;
    for ( uint64_t chart_id : chart_ids_ ) by_chart [ chart_id ] = {};
    for ( const TaggedDyadicCell & cell : cells ) {
      auto dimension = chart_dimensions_ . find ( cell . chart_id );
      if ( dimension == chart_dimensions_ . end () ) {
        std::ostringstream message;
        message << "AtlasModel active cell refers to unknown chart "
                << cell . chart_id;
        throw std::invalid_argument ( message . str () );
      }
      if ( cell . coordinates . size () != dimension -> second ) {
        std::ostringstream message;
        message << "AtlasModel active cell in chart " << cell . chart_id
                << " needs " << dimension -> second
                << " coordinate indices; got " << cell . coordinates . size ();
        throw std::invalid_argument ( message . str () );
      }
      by_chart [ cell . chart_id ] . push_back (
        AtlasDyadicCell ( cell . axis_depth, cell . coordinates ) );
    }

    std::shared_ptr<Atlas> candidate (
      dynamic_cast<Atlas *> ( phase_space_ -> clone () ) );
    for ( uint64_t chart_id : chart_ids_ ) {
      candidate -> set_chart_active_dyadic_cells (
        chart_id, by_chart [ chart_id ] );
    }

    std::vector<TaggedDyadicCell> canonical = cells;
    std::sort (
      canonical . begin (), canonical . end (),
      [] ( const TaggedDyadicCell & first, const TaggedDyadicCell & second ) {
        if ( first . chart_id != second . chart_id )
          return first . chart_id < second . chart_id;
        if ( first . axis_depth != second . axis_depth )
          return first . axis_depth < second . axis_depth;
        return first . coordinates < second . coordinates;
      } );
    canonical . erase (
      std::unique (
        canonical . begin (), canonical . end (),
        [] ( const TaggedDyadicCell & first, const TaggedDyadicCell & second ) {
          return first . chart_id == second . chart_id and
                 first . axis_depth == second . axis_depth and
                 first . coordinates == second . coordinates;
        } ),
      canonical . end () );

    phase_space_ = std::move ( candidate );
    active_dyadic_cells_ = std::move ( canonical );
    active_subgrid_configured_ = true;
  }

  std::shared_ptr<Grid> phaseSpace ( void ) const {
    if ( chart_dimensions_ . empty () ) {
      throw std::logic_error ( "AtlasModel has no charts" );
    }
    return std::shared_ptr<Grid> ( phase_space_ -> clone () );
  }

  std::shared_ptr<const Map> map ( void ) const {
    if ( not map_ ) {
      throw std::logic_error ( "AtlasModel has no map; call set_map first" );
    }
    return map_;
  }

  int phase_subdiv_min ( void ) const { return phase_subdiv_min_; }
  int phase_subdiv_max ( void ) const { return phase_subdiv_max_; }
  int phase_subdiv_init ( void ) const { return phase_subdiv_init_; }
  int phase_subdiv_limit ( void ) const { return phase_subdiv_limit_; }
  std::vector<uint64_t> chart_ids ( void ) const { return chart_ids_; }
  bool active_subgrid_configured ( void ) const {
    return active_subgrid_configured_;
  }
  std::vector<TaggedDyadicCell> active_dyadic_cells ( void ) const {
    return active_dyadic_cells_;
  }
  uint64_t initial_cell_count ( void ) const { return phase_space_ -> size (); }

private:
  int phase_subdiv_min_;
  int phase_subdiv_max_;
  int phase_subdiv_init_;
  int phase_subdiv_limit_;
  std::shared_ptr<Atlas> phase_space_;
  std::unordered_map<uint64_t, uint64_t> chart_dimensions_;
  std::vector<uint64_t> chart_ids_;
  std::shared_ptr<Map> map_;
  bool active_subgrid_configured_ = false;
  std::vector<TaggedDyadicCell> active_dyadic_cells_;
};

/// Python bindings

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace cmgdb_atlas_detail {

inline TaggedRectangle
parse_tagged_rectangle ( py::handle value ) {
  if ( py::isinstance<TaggedRectangle> ( value ) ) {
    return py::cast<TaggedRectangle> ( value );
  }

  py::handle chart_value;
  py::handle bounds_value;
  if ( py::isinstance<py::dict> ( value ) ) {
    py::dict item = py::reinterpret_borrow<py::dict> ( value );
    if ( item . contains ( "chart_id" ) ) {
      chart_value = item [ "chart_id" ];
    } else if ( item . contains ( "chart" ) ) {
      chart_value = item [ "chart" ];
    } else {
      throw py::value_error (
        "tagged image-piece dictionary needs 'chart_id' (or 'chart')" );
    }
    if ( not item . contains ( "bounds" ) ) {
      throw py::value_error (
        "tagged image-piece dictionary needs 'bounds'" );
    }
    bounds_value = item [ "bounds" ];
  } else if ( py::hasattr ( value, "chart_id" ) and
              py::hasattr ( value, "bounds" ) ) {
    chart_value = value . attr ( "chart_id" );
    bounds_value = value . attr ( "bounds" );
  } else {
    py::sequence pair;
    try {
      pair = py::reinterpret_borrow<py::sequence> ( value );
      if ( py::len ( pair ) != 2 ) throw py::value_error (
        "tagged image piece must be (chart_id, bounds)" );
    } catch ( const py::error_already_set & ) {
      throw py::value_error (
        "tagged image piece must be (chart_id, bounds), a dictionary, or TaggedRectangle" );
    }
    chart_value = pair [ 0 ];
    bounds_value = pair [ 1 ];
  }

  try {
    return TaggedRectangle (
      py::cast<uint64_t> ( chart_value ),
      py::cast<std::vector<double>> ( bounds_value ) );
  } catch ( const py::cast_error & ) {
    throw py::value_error (
      "tagged image piece needs a nonnegative integer chart_id and numeric bounds" );
  }
}

inline TaggedDyadicCell
parse_tagged_dyadic_cell ( py::handle value ) {
  if ( py::isinstance<TaggedDyadicCell> ( value ) ) {
    return py::cast<TaggedDyadicCell> ( value );
  }

  py::handle chart_value;
  py::handle depth_value;
  py::handle coordinates_value;
  if ( py::isinstance<py::dict> ( value ) ) {
    py::dict item = py::reinterpret_borrow<py::dict> ( value );
    if ( not item . contains ( "chart_id" ) or
         not item . contains ( "axis_depth" ) or
         not item . contains ( "coordinates" ) ) {
      throw py::value_error (
        "active dyadic cell dictionary needs chart_id, axis_depth, and coordinates" );
    }
    chart_value = item [ "chart_id" ];
    depth_value = item [ "axis_depth" ];
    coordinates_value = item [ "coordinates" ];
  } else if ( py::hasattr ( value, "chart_id" ) and
              py::hasattr ( value, "axis_depth" ) and
              py::hasattr ( value, "coordinates" ) ) {
    chart_value = value . attr ( "chart_id" );
    depth_value = value . attr ( "axis_depth" );
    coordinates_value = value . attr ( "coordinates" );
  } else {
    try {
      py::sequence item = py::reinterpret_borrow<py::sequence> ( value );
      if ( py::len ( item ) != 3 ) throw py::value_error (
        "active dyadic cell must be (chart_id, axis_depth, coordinates)" );
      chart_value = item [ 0 ];
      depth_value = item [ 1 ];
      coordinates_value = item [ 2 ];
    } catch ( const py::error_already_set & ) {
      throw py::value_error (
        "active dyadic cell must be a triple, dictionary, or TaggedDyadicCell" );
    }
  }

  try {
    return TaggedDyadicCell (
      py::cast<uint64_t> ( chart_value ),
      py::cast<uint64_t> ( depth_value ),
      py::cast<std::vector<uint64_t>> ( coordinates_value ) );
  } catch ( const py::cast_error & ) {
    throw py::value_error (
      "active dyadic cell needs nonnegative integer chart_id, axis_depth, and coordinates" );
  }
}

inline AtlasModel::Callback
wrap_python_callback ( py::function callback ) {
  return [ callback ] ( uint64_t chart_id,
                        std::vector<double> bounds ) {
    py::gil_scoped_acquire acquire;
    py::object raw_pieces = callback ( chart_id, std::move ( bounds ) );
    if ( raw_pieces . is_none () ) {
      throw py::value_error (
        "AtlasModel map callback must return an iterable (use [] for no image)" );
    }

    std::vector<TaggedRectangle> result;
    try {
      py::iterable pieces =
        py::reinterpret_borrow<py::iterable> ( raw_pieces );
      for ( py::handle piece : pieces ) {
        result . push_back ( parse_tagged_rectangle ( piece ) );
      }
    } catch ( const py::error_already_set & error ) {
      if ( error . matches ( PyExc_TypeError ) ) {
        throw py::value_error (
          "AtlasModel map callback must return an iterable of tagged image pieces" );
      }
      throw;
    }
    return result;
  };
}

inline TaggedRectangle
atlas_cell ( const Atlas & atlas, uint64_t cell ) {
  if ( cell >= atlas . size () ) {
    throw py::index_error ( "Atlas cell index is out of range" );
  }
  std::shared_ptr<AtlasGeo> geo =
    std::dynamic_pointer_cast<AtlasGeo> ( atlas . geometry ( cell ) );
  const RectGeo & rectangle = geo -> rect ();
  const uint64_t dimension = rectangle . dimension ();
  std::vector<double> bounds ( 2 * dimension );
  for ( uint64_t d = 0; d < dimension; ++ d ) {
    bounds [ d ] = rectangle . lower_bounds [ d ];
    bounds [ dimension + d ] = rectangle . upper_bounds [ d ];
  }
  return TaggedRectangle ( geo -> id (), std::move ( bounds ) );
}

inline std::vector<uint64_t>
atlas_cover ( const Atlas & atlas,
              uint64_t chart_id,
              const std::vector<double> & bounds ) {
  uint64_t dimension = 0;
  bool found = false;
  for ( const Atlas::IdChartPair & pair : atlas . charts () ) {
    if ( pair . first == chart_id ) {
      dimension = pair . second -> dimension ();
      found = true;
      break;
    }
  }
  if ( not found ) {
    std::ostringstream message;
    message << "Atlas.cover received unknown chart " << chart_id;
    throw py::value_error ( message . str () );
  }
  if ( bounds . size () != 2 * dimension ) {
    std::ostringstream message;
    message << "Atlas.cover expected " << 2 * dimension
            << " bounds for chart " << chart_id
            << "; got " << bounds . size ();
    throw py::value_error ( message . str () );
  }
  std::vector<double> lower ( bounds . begin (), bounds . begin () + dimension );
  std::vector<double> upper ( bounds . begin () + dimension, bounds . end () );
  for ( uint64_t d = 0; d < dimension; ++ d ) {
    if ( not std::isfinite ( lower [ d ] ) or
         not std::isfinite ( upper [ d ] ) or
         lower [ d ] > upper [ d ] ) {
      throw py::value_error (
        "Atlas.cover requires finite bounds with lower <= upper" );
    }
  }
  AtlasGeo target (
    chart_id, RectGeo ( dimension, std::move ( lower ), std::move ( upper ) ) );
  return atlas . cover ( target );
}

} // namespace cmgdb_atlas_detail

inline void
AtlasModelBinding ( py::module & m ) {
  py::class_<TaggedRectangle> ( m, "TaggedRectangle" )
    .def ( py::init<uint64_t, std::vector<double>> (),
           py::arg ( "chart_id" ), py::arg ( "bounds" ) )
    .def_readwrite ( "chart_id", &TaggedRectangle::chart_id )
    .def_readwrite ( "bounds", &TaggedRectangle::bounds );

  py::class_<TaggedDyadicCell> ( m, "TaggedDyadicCell" )
    .def ( py::init<uint64_t, uint64_t, std::vector<uint64_t>> (),
           py::arg ( "chart_id" ), py::arg ( "axis_depth" ),
           py::arg ( "coordinates" ) )
    .def_readwrite ( "chart_id", &TaggedDyadicCell::chart_id )
    .def_readwrite ( "axis_depth", &TaggedDyadicCell::axis_depth )
    .def_readwrite ( "coordinates", &TaggedDyadicCell::coordinates );

  py::class_<Atlas, Grid, std::shared_ptr<Atlas>> ( m, "Atlas" )
    .def ( "num_charts", &Atlas::numCharts )
    .def ( "cell", &cmgdb_atlas_detail::atlas_cell,
           py::arg ( "index" ),
           "Return the chart id and rectangle bounds for an atlas cell." )
    .def ( "cover", &cmgdb_atlas_detail::atlas_cover,
           py::arg ( "chart_id" ), py::arg ( "bounds" ),
           "Return active Atlas cells intersecting one tagged rectangle; an "
           "empty result is an explicit active-boundary exit." );

  py::class_<AtlasModel, std::shared_ptr<AtlasModel>> ( m, "AtlasModel" )
    .def ( py::init<int> (), py::arg ( "phase_subdiv" ) )
    .def ( py::init<int, int, int, int> (),
           py::arg ( "phase_subdiv_min" ),
           py::arg ( "phase_subdiv_max" ),
           py::arg ( "phase_subdiv_init" ),
           py::arg ( "phase_subdiv_limit" ) = 10000 )
    .def ( "add_chart",
           py::overload_cast<uint64_t, const std::vector<double> &,
                             const std::vector<double> &>
             ( &AtlasModel::add_chart ),
           py::arg ( "chart_id" ), py::arg ( "lower_bounds" ),
           py::arg ( "upper_bounds" ) )
    .def ( "add_chart",
           py::overload_cast<uint64_t, const std::vector<double> &,
                             const std::vector<double> &,
                             const std::vector<bool> &>
             ( &AtlasModel::add_chart ),
           py::arg ( "chart_id" ), py::arg ( "lower_bounds" ),
           py::arg ( "upper_bounds" ), py::arg ( "periodic" ) )
    .def ( "set_map",
           [] ( AtlasModel & model, py::function callback ) {
             model . set_map (
               cmgdb_atlas_detail::wrap_python_callback (
                 std::move ( callback ) ) );
           },
           py::arg ( "callback" ),
           "Set (chart_id, rectangle) -> finite tagged-union rectangle map." )
    .def ( "set_active_subgrid",
           [] ( AtlasModel & model, py::iterable raw_cells ) {
             std::vector<TaggedDyadicCell> cells;
             for ( py::handle value : raw_cells ) {
               cells . push_back (
                 cmgdb_atlas_detail::parse_tagged_dyadic_cell ( value ) );
             }
             model . set_active_subgrid ( cells );
           },
           py::arg ( "cells" ),
           "Set the initial active Atlas grid from tagged dyadic chart cells. "
           "Omitted charts are empty; targets outside the active family are exits." )
    .def ( "phaseSpace", &AtlasModel::phaseSpace )
    .def ( "phase_subdiv_min", &AtlasModel::phase_subdiv_min )
    .def ( "phase_subdiv_max", &AtlasModel::phase_subdiv_max )
    .def ( "phase_subdiv_init", &AtlasModel::phase_subdiv_init )
    .def ( "phase_subdiv_limit", &AtlasModel::phase_subdiv_limit )
    .def ( "chart_ids", &AtlasModel::chart_ids )
    .def ( "active_subgrid_configured", &AtlasModel::active_subgrid_configured )
    .def ( "active_dyadic_cells", &AtlasModel::active_dyadic_cells )
    .def ( "initial_cell_count", &AtlasModel::initial_cell_count );
}

#endif
