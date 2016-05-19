#ifndef SimMuon_GEMDigitizer_ME0DigiPreRecoProducer_h
#define SimMuon_GEMDigitizer_ME0DigiPreRecoProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/ME0DigiSimLink/interface/ME0DigiSimLink.h"

#include <string>

class ME0Geometry;
class ME0DigiPreRecoModel;

class ME0DigiPreRecoProducer : public edm::stream::EDProducer<>
{
public:

  typedef edm::DetSetVector<ME0DigiSimLink> ME0DigiSimLinks;

  explicit ME0DigiPreRecoProducer(const edm::ParameterSet& ps);

  virtual ~ME0DigiPreRecoProducer();

  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;

  virtual void produce(edm::Event&, const edm::EventSetup&) override;

private:

  //Name of Collection used for create the XF 
  edm::EDGetTokenT<CrossingFrame<PSimHit> > cf_token; 

  std::string digiPreRecoModelString_;
  ME0DigiPreRecoModel* me0DigiPreRecoModel_;
};

#endif

