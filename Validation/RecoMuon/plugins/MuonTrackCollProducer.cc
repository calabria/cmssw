
#include "Validation/RecoMuon/plugins/MuonTrackCollProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include <sstream>

bool MuonTrackCollProducer::isLoose(edm::Event& iEvent, reco::MuonCollection::const_iterator muon)
{
  bool isPF = muon->isPFMuon();
  bool isGLB = muon->isGlobalMuon();
  bool isTrk = muon->isTrackerMuon();

  return (isPF && (isGLB || isTrk) );
}

bool MuonTrackCollProducer::isSoft(edm::Event& iEvent, reco::MuonCollection::const_iterator muon, bool useIPxy, bool useIPz)
{
  bool result = false;

  if (muon->muonBestTrack().isNonnull() && muon->innerTrack().isNonnull()){

 	edm::Handle<reco::VertexCollection> vertexHandle;
  	iEvent.getByLabel(vxtTag,vertexHandle);
  	const reco::VertexCollection* vertexes = vertexHandle.product();

	bool isGood = muon::isGoodMuon((*muon), muon::TMOneStationTight); 
	bool trkLayMeas = muon->muonBestTrack()->hitPattern().trackerLayersWithMeasurement() > 5; 
	bool pxlLayMeas = muon->innerTrack()->hitPattern().pixelLayersWithMeasurement() > 0; 
	bool quality = muon->innerTrack()->quality(reco::Track::highPurity);
	bool ipxy = false;
	bool ipz = false;
	if(vertexes->size()!=0 && useIPxy) ipxy = fabs(muon->muonBestTrack()->dxy((*vertexes)[0].position())) < 0.2;
	else ipxy = true;
 	if(vertexes->size()!=0 && useIPz) ipz = fabs(muon->muonBestTrack()->dz((*vertexes)[0].position())) < 0.5;
	else ipz = true;
	if(isGood && trkLayMeas && pxlLayMeas && quality && ipxy && ipz) result = true;

  }

  return result;
}

bool MuonTrackCollProducer::isTight(edm::Event& iEvent, reco::MuonCollection::const_iterator muon, bool useIPxy, bool useIPz)
{
  bool result = false;

  if (muon->muonBestTrack().isNonnull() && muon->innerTrack().isNonnull() && muon->globalTrack().isNonnull()){
		
 	edm::Handle<reco::VertexCollection> vertexHandle;
  	iEvent.getByLabel(vxtTag,vertexHandle);
  	const reco::VertexCollection* vertexes = vertexHandle.product();

	bool trkLayMeas = muon->muonBestTrack()->hitPattern().trackerLayersWithMeasurement() > 5; 
	bool isGlb = muon->isGlobalMuon(); 
	bool isPF = muon->isPFMuon(); 
	bool chi2 = muon->globalTrack()->normalizedChi2() < 10.; 
	bool validHits = muon->globalTrack()->hitPattern().numberOfValidMuonHits() > 0; 
	bool matchedSt = muon->numberOfMatchedStations() > 1; 
	bool ipxy = false;
	bool ipz = false;
	if(vertexes->size()!=0 && useIPxy) ipxy = fabs(muon->muonBestTrack()->dxy((*vertexes)[0].position())) < 0.2;
	else ipxy = true;
 	if(vertexes->size()!=0 && useIPz) ipz = fabs(muon->muonBestTrack()->dz((*vertexes)[0].position())) < 0.5;
	else ipz = true;
	bool validPxlHit = muon->innerTrack()->hitPattern().numberOfValidPixelHits() > 0;

	if(trkLayMeas && isGlb && isPF && chi2 && validHits && matchedSt && ipxy && ipz && validPxlHit) result = true;

  }

  return result;
}

MuonTrackCollProducer::MuonTrackCollProducer(const edm::ParameterSet& parset) :
  muonsTag(parset.getParameter< edm::InputTag >("muonsTag")),
  vxtTag(parset.getParameter< edm::InputTag >("vxtTag")),
  useIPxy(parset.getUntrackedParameter< bool >("useIPxy", true)),
  useIPz(parset.getUntrackedParameter< bool >("useIPz", true)),
  selectionTags(parset.getParameter< std::vector<std::string> >("selectionTags")),
  trackType(parset.getParameter< std::string >("trackType")),
  parset_(parset)
{
  produces<reco::TrackCollection>();
  //produces<reco::TrackExtraCollection>();
  //produces<TrackingRecHitCollection>();
}

MuonTrackCollProducer::~MuonTrackCollProducer() {
}

void MuonTrackCollProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  iEvent.getByLabel(muonsTag,muonCollectionH);
  
  std::auto_ptr<reco::TrackCollection> selectedTracks(new reco::TrackCollection);
  //std::auto_ptr<reco::TrackExtraCollection> selectedTrackExtras( new reco::TrackExtraCollection() );
  //std::auto_ptr<TrackingRecHitCollection> selectedTrackHits( new TrackingRecHitCollection() );

  reco::TrackRefProd rTracks = iEvent.getRefBeforePut<reco::TrackCollection>();
  //reco::TrackExtraRefProd rTrackExtras = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
  //TrackingRecHitRefProd rHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();

  //edm::Ref<reco::TrackExtraCollection>::key_type idx = 0;
  //edm::Ref<reco::TrackExtraCollection>::key_type hidx = 0;

  edm::LogVerbatim("MuonTrackProducer") <<"\nThere are "<< muonCollectionH->size() <<" reco::Muons.";
  unsigned int muon_index = 0;
  for(reco::MuonCollection::const_iterator muon = muonCollectionH->begin();
       muon != muonCollectionH->end(); ++muon, muon_index++) {
    edm::LogVerbatim("MuonTrackProducer") <<"\n******* muon index : "<<muon_index;
    
    std::vector<bool> isGood;
    for(unsigned int index=0; index<selectionTags.size(); ++index) {
      isGood.push_back(false);

      muon::SelectionType muonType = muon::selectionTypeFromString(selectionTags[index]);
      isGood[index] = muon::isGoodMuon(*muon, muonType);
    }

    bool isGoodResult=true;
    for(unsigned int index=0; index<isGood.size(); ++index) {
      edm::LogVerbatim("MuonTrackProducer") << "selectionTag = "<<selectionTags[index]<< ": "<<isGood[index]<<"\n";
      isGoodResult *= isGood[index];
    }

    if (isGoodResult) {
      // new copy of Track
      reco::TrackRef trackref;
      bool loose = isLoose(iEvent, muon);
      bool soft = isSoft(iEvent, muon, useIPxy, useIPz);
      bool tight = isTight(iEvent, muon, useIPxy, useIPz);
      if (trackType == "innerTrack") {
        if (muon->innerTrack().isNonnull()) trackref = muon->innerTrack();
        else continue;
      }
      else if (trackType == "outerTrack") {
        if (muon->outerTrack().isNonnull()) trackref = muon->outerTrack();
        else continue;
      }
      else if (trackType == "globalTrack") {
        if (muon->globalTrack().isNonnull()) trackref = muon->globalTrack();
        else continue;
      }
      else if (trackType == "innerTrackPlusSegments") {
	if (muon->innerTrack().isNonnull()) trackref = muon->innerTrack();
	else continue;
      }
      else if (trackType == "bestMuon") {
	if (muon->muonBestTrack().isNonnull()) trackref = muon->muonBestTrack();
	else continue;
      }
      else if (trackType == "bestMuonLoose") {
	if (muon->muonBestTrack().isNonnull() && loose) trackref = muon->muonBestTrack();
	else continue;
      }
      else if (trackType == "bestMuonSoft") {
	if (muon->muonBestTrack().isNonnull() && soft) trackref = muon->muonBestTrack();
	else continue;
      }
      else if (trackType == "bestMuonTight") {
	if (muon->muonBestTrack().isNonnull() && tight) trackref = muon->muonBestTrack();
	else continue;
      }
      else if (trackType == "bestMuonTuneP") {
	if (muon->tunePMuonBestTrack().isNonnull()) trackref = muon->tunePMuonBestTrack();
	else continue;
      }
      else if (trackType == "cocktailBest") {

  	//None, InnerTrack, OuterTrack, CombinedTrack,TPFMS, Picky, DYT
	reco::Muon::MuonTrackType type = muon->muonBestTrackType();

	switch((int) type){

		case 1: 
			if (muon->innerTrack().isNonnull()) trackref = muon->innerTrack();
			else continue;
			break;
		case 3:
			if (muon->globalTrack().isNonnull()) trackref = muon->globalTrack();
			else continue;
			break;
		case 4:
			if (muon->tpfmsTrack().isNonnull()) trackref = muon->tpfmsTrack();
			else continue;
			break;
		case 5:
			if (muon->pickyTrack().isNonnull()) trackref = muon->pickyTrack();
			else continue;
			break;
		default: continue;

	}

      }
      else if (trackType == "cocktailTuneP") {

  	//None, InnerTrack, OuterTrack, CombinedTrack,TPFMS, Picky, DYT
	reco::Muon::MuonTrackType type = muon->tunePMuonBestTrackType();
	
	switch((int) type){

		case 1: 
			if (muon->innerTrack().isNonnull()) trackref = muon->innerTrack();
			else continue;
			break;
		case 3:
			if (muon->globalTrack().isNonnull()) trackref = muon->globalTrack();
			else continue;
			break;
		case 4:
			if (muon->tpfmsTrack().isNonnull()) trackref = muon->tpfmsTrack();
			else continue;
			break;
		case 5:
			if (muon->pickyTrack().isNonnull()) trackref = muon->pickyTrack();
			else continue;
			break;
		default: continue;

	}

      }

      const reco::Track* trk = &(*trackref);
      // pointer to old track:
      reco::Track* newTrk = new reco::Track(*trk);

      //newTrk->setExtra( reco::TrackExtraRef( rTrackExtras, idx++ ) );
      //PropagationDirection seedDir = trk->seedDirection();
      // new copy of track Extras
      /*reco::TrackExtra * newExtra = new reco::TrackExtra( trk->outerPosition(), trk->outerMomentum(),
                                        trk->outerOk(), trk->innerPosition(),
                                        trk->innerMomentum(), trk->innerOk(),
                                        trk->outerStateCovariance(), trk->outerDetId(),
                                        trk->innerStateCovariance(), trk->innerDetId() , seedDir ) ;*/

      // new copy of the silicon hits; add hit refs to Extra and hits to hit collection
      //unsigned int index_hit = 0;
      
      // edm::LogVerbatim("MuonTrackProducer")<<"\n printing initial hit_pattern";
      // trk->hitPattern().print();

      /*for (trackingRecHit_iterator iHit = trk->recHitsBegin(); iHit != trk->recHitsEnd(); iHit++) {
        TrackingRecHit* hit = (*iHit)->clone();
        index_hit++;
        selectedTrackHits->push_back( hit );

        newExtra->add( TrackingRecHitRef( rHits, hidx++ ) );
      }*/

      // edm::LogVerbatim("MuonTrackProducer")<<"\n printing final hit_pattern";
      // newTrk->hitPattern().print();
      
      selectedTracks->push_back( *newTrk );
      //selectedTrackExtras->push_back( *newExtra );

    } // if (isGoodResult)
  } // loop on reco::MuonCollection
  
  iEvent.put(selectedTracks);
  //iEvent.put(selectedTrackExtras);
  //iEvent.put(selectedTrackHits);
}
