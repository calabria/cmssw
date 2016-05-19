#ifndef SimMuon_GEMDigitizer_ME0DigiPreRecoModel_h
#define SimMuon_GEMDigitizer_ME0DigiPreRecoModel_h

/** 
 *  \class ME0DigiPreRecoModel
 *
 *  Base Class for the ME0 strip response simulation 
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/ME0DigiSimLink/interface/ME0DigiSimLink.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

#include <map>
#include <set>

namespace CLHEP {
  class HepRandomEngine;
}

class ME0EtaPartition;
class ME0Geometry;
class PSimHit;

class ME0DigiPreRecoModel
{
public:
    
  typedef edm::DetSet<ME0DigiSimLink> ME0DigiSimLinks;
  
  virtual ~ME0DigiPreRecoModel() {}

  void setGeometry(const ME0Geometry *geom) {geometry_ = geom;}

  const ME0Geometry* getGeometry() {return geometry_;}

  virtual void simulateSignal(const ME0EtaPartition*, const edm::PSimHitContainer&, CLHEP::HepRandomEngine* engine) = 0;

  virtual void simulateNoise(const ME0EtaPartition*, CLHEP::HepRandomEngine* engine) = 0;
  
  //  virtual std::vector<std::pair<int,int> > 
  //   simulateClustering(const ME0EtaPartition*, const PSimHit*, const int) = 0;

  void fillDigis(int rollDetId, ME0DigiPreRecoCollection&);

  virtual void setup() = 0;
    
  const ME0DigiSimLinks & me0DigiSimLinks() const {return theMe0DigiSimLinks_;}

protected:
  std::set< ME0DigiPreReco > digi_;
    
  void addLinksWithPartId(float x, float y, float ex, float ey, float corr, float tof);

  ME0DigiPreRecoModel(const edm::ParameterSet&) {}

  const ME0Geometry * geometry_;
    
  typedef std::multimap<
      ME0DigiPreReco,
      const PSimHit*,
      std::less< ME0DigiPreReco >
    >  DetectorHitMap;
  
  DetectorHitMap detectorHitMap_;
  ME0DigiSimLinks theMe0DigiSimLinks_;
};
#endif
