#ifndef CMDB_MAP_H
#define CMDB_MAP_H

#include <memory>
#include <vector>
#include "Geo.h"

class Map {
public:
  virtual ~Map ( void ) {}
  virtual std::shared_ptr<Geo> operator () ( std::shared_ptr<Geo> geo ) const = 0;

  virtual std::vector<std::shared_ptr<Geo>>
  batch_map ( const std::vector<std::shared_ptr<Geo>>& geos ) const {
    std::vector<std::shared_ptr<Geo>> results;
    results.reserve(geos.size());
    for ( const auto& geo : geos ) {
      results.push_back(operator()(geo));
    }
    return results;
  }

  virtual bool has_optimized_batch ( void ) const {
    return false;
  }

  virtual bool is_thread_safe ( void ) const {
    return false;
  }
private:
};

#endif
