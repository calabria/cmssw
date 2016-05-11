//
// modified & integrated by Giovanni Abbiendi
// from code by Arun Luthra: UserCode/luthra/MuonTrackSelector/src/MuonTrackSelector.cc
//
#ifndef MCTruth_MuonTrackProducer_h
#define MCTruth_MuonTrackProducer_h

#include <memory>
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

class MuonTrackProducer : public edm::EDProducer {
  public:
    explicit MuonTrackProducer(const edm::ParameterSet&);
     std::vector<double> findSimVtx(edm::Event& iEvent);
     bool isGlobalTightMuon(const reco::MuonCollection::const_iterator muonRef);
     bool isTrackerTightMuon(const reco::MuonCollection::const_iterator muonRef);
     bool isIsolatedMuon(const reco::MuonCollection::const_iterator muonRef);
     bool isLoose(edm::Event& iEvent, reco::MuonCollection::const_iterator muon);
     bool isLooseMod(edm::Event& iEvent, reco::MuonCollection::const_iterator muon);
     bool isSoft(edm::Event& iEvent, reco::MuonCollection::const_iterator muon, bool useIPxy, bool useIPz);
     bool isTight(edm::Event& iEvent, reco::MuonCollection::const_iterator muon, bool useIPxy, bool useIPz);
     bool isTightBS(edm::Event& iEvent, reco::MuonCollection::const_iterator muon);
     bool isTightMod(edm::Event& iEvent, reco::MuonCollection::const_iterator muon, bool useIPxy, bool useIPz);
     bool isLoose2(edm::Event& iEvent, reco::MuonCollection::const_iterator muon, bool useIPxy, bool useIPz);
    ~MuonTrackProducer();

  private:
    virtual void produce(edm::Event&, const edm::EventSetup&);
  
    edm::Handle<reco::MuonCollection> muonCollectionH;
    edm::Handle<DTRecSegment4DCollection> dtSegmentCollectionH_;
    edm::Handle<CSCSegmentCollection> cscSegmentCollectionH_;

    edm::InputTag muonsTag;
    edm::InputTag vxtTag;
    bool useIPxy, useIPz;
    edm::InputTag inputDTRecSegment4DCollection_;
    edm::InputTag inputCSCSegmentCollection_;
    std::vector<std::string> selectionTags;
    std::string trackType;
    const edm::ParameterSet parset_;
};

#endif
