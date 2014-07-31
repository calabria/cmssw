
#ifndef RecoMuon_MuonTrackCollProducer_h
#define RecoMuon_MuonTrackCollProducer_h

#include <memory>
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

class MuonTrackCollProducer : public edm::EDProducer {
  public:
    explicit MuonTrackCollProducer(const edm::ParameterSet&);
     bool isLoose(edm::Event& iEvent, reco::MuonCollection::const_iterator muon);
     bool isSoft(edm::Event& iEvent, reco::MuonCollection::const_iterator muon, bool useIP);
     bool isTight(edm::Event& iEvent, reco::MuonCollection::const_iterator muon, bool useIP);
    ~MuonTrackCollProducer();

  private:
    virtual void produce(edm::Event&, const edm::EventSetup&);
  
    edm::Handle<reco::MuonCollection> muonCollectionH;
    edm::InputTag muonsTag;
    edm::InputTag vxtTag;
    bool useIP;
    std::vector<std::string> selectionTags;
    std::string trackType;
    const edm::ParameterSet parset_;
};

#endif
