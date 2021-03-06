#ifndef SimCalorimetry_HGCSimProducers_hgceedigitizer
#define SimCalorimetry_HGCSimProducers_hgceedigitizer

#include "SimCalorimetry/HGCSimProducers/interface/HGCDigitizerBase.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

class HGCEEDigitizer : public HGCDigitizerBase<HGCEEDataFrame>
{
 public:
  HGCEEDigitizer(const edm::ParameterSet& ps);
  void runDigitizer(std::auto_ptr<HGCEEDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData);
  ~HGCEEDigitizer();
 private:

};

#endif 
