#include "SimMuon/GEMDigitizer/interface/ME0DigiPreRecoModel.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

void 
ME0DigiPreRecoModel::fillDigis(int rollDetId, ME0DigiPreRecoCollection& digis)
{
  for (auto d: digi_)
  {
    digis.insertDigi(ME0DetId(rollDetId), d);
    addLinksWithPartId(d.x(), d.y(), d.ex(), d.ey(), d.corr(), d.tof());
  }
  digi_.clear();
}

void ME0DigiPreRecoModel::addLinksWithPartId(float x, float y, float ex, float ey, float corr, float tof)
{
    
    ME0DigiPreReco digi(x, y, ex, ey, corr, tof, 1);
    std::pair<DetectorHitMap::iterator, DetectorHitMap::iterator> channelHitItr
    = detectorHitMap_.equal_range(digi);
    
    for( DetectorHitMap::iterator hitItr = channelHitItr.first; hitItr != channelHitItr.second; ++hitItr)
    {
        const PSimHit * hit = (hitItr->second);
        // might be zero for unit tests and such
        if (hit == nullptr) continue;
        
        theMe0DigiSimLinks_.push_back(ME0DigiSimLink(digi, hit->entryPoint(), hit->momentumAtEntry(), hit->timeOfFlight(), hit->energyLoss(), hit->particleType(), hit->detUnitId(), hit->trackId(), hit->eventId(), hit->processType()));
        
    }
}