#ifndef CMDB_ATLAS_MODEL_MAP_F_H
#define CMDB_ATLAS_MODEL_MAP_F_H

#include "AtlasGeo.h"
#include "Map.h"
#include "RectGeo.h"
#include "UnionGeo.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

/** A rectangle tagged by the Atlas chart in which it lives. */
struct TaggedRectangle {
  uint64_t chart_id;
  std::vector<double> bounds;

  TaggedRectangle ( void ) : chart_id ( 0 ) {}
  TaggedRectangle ( uint64_t chart,
                    std::vector<double> rectangle_bounds )
    : chart_id ( chart ), bounds ( std::move ( rectangle_bounds ) ) {}
};

/** Python-backed finite-union map for an Atlas.
 *
 * The callback receives the source chart id and the source rectangle bounds.
 * It returns zero or more target rectangles, each carrying its target chart
 * id.  The image is represented by UnionGeo, so Grid::cover covers each piece
 * separately instead of first replacing the pieces by their Euclidean hull.
 */
class AtlasModelMapF : public Map {
public:
  typedef std::function<
    std::vector<TaggedRectangle> ( uint64_t, std::vector<double> )
  > Callback;

  AtlasModelMapF (
      std::unordered_map<uint64_t, uint64_t> chart_dimensions,
      Callback callback )
    : chart_dimensions_ ( std::move ( chart_dimensions ) ),
      callback_ ( std::move ( callback ) ) {}

  std::shared_ptr<Geo>
  operator () ( std::shared_ptr<Geo> geo ) const override {
    std::shared_ptr<AtlasGeo> source =
      std::dynamic_pointer_cast<AtlasGeo> ( geo );
    if ( not source ) {
      throw std::invalid_argument (
        "AtlasModel map expected an AtlasGeo source cell" );
    }

    const RectGeo & source_rect = source -> rect ();
    const uint64_t source_dimension = source_rect . dimension ();
    std::vector<double> source_bounds ( 2 * source_dimension, 0.0 );
    for ( uint64_t d = 0; d < source_dimension; ++ d ) {
      source_bounds [ d ] = source_rect . lower_bounds [ d ];
      source_bounds [ source_dimension + d ] =
        source_rect . upper_bounds [ d ];
    }

    const std::vector<TaggedRectangle> pieces =
      callback_ ( source -> id (), std::move ( source_bounds ) );
    std::shared_ptr<UnionGeo> image ( new UnionGeo );

    for ( size_t piece_index = 0;
          piece_index < pieces . size ();
          ++ piece_index ) {
      const TaggedRectangle & piece = pieces [ piece_index ];
      const auto dimension_it = chart_dimensions_ . find ( piece . chart_id );
      if ( dimension_it == chart_dimensions_ . end () ) {
        std::ostringstream message;
        message << "AtlasModel map piece " << piece_index
                << " targets unknown chart " << piece . chart_id;
        throw std::invalid_argument ( message . str () );
      }

      const uint64_t target_dimension = dimension_it -> second;
      if ( piece . bounds . size () != 2 * target_dimension ) {
        std::ostringstream message;
        message << "AtlasModel map piece " << piece_index << " in chart "
                << piece . chart_id << " has " << piece . bounds . size ()
                << " bounds; expected " << 2 * target_dimension;
        throw std::invalid_argument ( message . str () );
      }

      RectGeo target_rect ( target_dimension );
      for ( uint64_t d = 0; d < target_dimension; ++ d ) {
        const double lower = piece . bounds [ d ];
        const double upper = piece . bounds [ target_dimension + d ];
        if ( not std::isfinite ( lower ) or not std::isfinite ( upper ) ) {
          std::ostringstream message;
          message << "AtlasModel map piece " << piece_index << " in chart "
                  << piece . chart_id << " has a non-finite bound";
          throw std::invalid_argument ( message . str () );
        }
        if ( lower > upper ) {
          std::ostringstream message;
          message << "AtlasModel map piece " << piece_index << " in chart "
                  << piece . chart_id << " has lower bound greater than upper "
                  << "bound in coordinate " << d;
          throw std::invalid_argument ( message . str () );
        }
        target_rect . lower_bounds [ d ] = lower;
        target_rect . upper_bounds [ d ] = upper;
      }

      image -> insert ( std::shared_ptr<Geo> (
        new AtlasGeo ( piece . chart_id, target_rect ) ) );
    }
    return image;
  }

private:
  std::unordered_map<uint64_t, uint64_t> chart_dimensions_;
  Callback callback_;
};

#endif
