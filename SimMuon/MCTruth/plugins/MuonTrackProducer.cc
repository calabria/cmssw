//
// modified & integrated by Giovanni Abbiendi
// from code by Arun Luthra: UserCode/luthra/MuonTrackSelector/src/MuonTrackSelector.cc
//
#include "SimMuon/MCTruth/plugins/MuonTrackProducer.h"
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
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include <sstream>

std::vector<double> MuonTrackProducer::findSimVtx(edm::Event& iEvent){

  edm::Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByLabel("genParticles", genParticles);
  std::vector<double> vtxCoord;
  vtxCoord.push_back(0);
  vtxCoord.push_back(0);
  vtxCoord.push_back(0);
  vtxCoord.push_back(0);
  vtxCoord.push_back(0);
  vtxCoord.push_back(0);
  vtxCoord.push_back(0);

  if(genParticles.isValid()){

  	for(reco::GenParticleCollection::const_iterator itg = genParticles->begin(); itg != genParticles->end(); ++itg ){

		int id = itg->pdgId();
		int status = itg->status();
		//int nDaughters = itg->numberOfDaughters();
		int nMothers = itg->numberOfMothers();
		//double phiGen = itg->phi();
		//double etaGen = itg->eta();
		//std::cout<<"id "<<id<<" "<<status<<" "<<nMothers<<" "<<phiGen<<" "<<etaGen<<std::endl;

		if((abs(id) == 23 || abs(id) == 22) && status == 3){

	 		vtxCoord[0] = 1;

			vtxCoord[4] = (double)(itg->vx()); 
			vtxCoord[5] = (double)(itg->vy());
			vtxCoord[6] = (double)(itg->vz());

		}

		//if(fabs(id) == 13) std::cout<<"ID "<<id<<" Status "<<status<<std::endl;

		else if(abs(id) == 13 && status == 1 && nMothers == 0){

	 		vtxCoord[0] = 2;

			vtxCoord[1] = (double)(itg->vx()); 
			vtxCoord[2] = (double)(itg->vy());
			vtxCoord[3] = (double)(itg->vz());

		}

		else if(abs(id) == 12 && status == 1 && nMothers == 0){

	 		vtxCoord[0] = 3;

			vtxCoord[1] = (double)(itg->vx()); 
			vtxCoord[2] = (double)(itg->vy());
			vtxCoord[3] = (double)(itg->vz());

		}

	}

  }

  //std::cout<<vtxCoord.size()<<" "<<vtxCoord[0]<<std::endl;
  return vtxCoord;

}

bool MuonTrackProducer::isLoose(edm::Event& iEvent, reco::MuonCollection::const_iterator muon)
{
  bool isPF = muon->isPFMuon();
  bool isGLB = muon->isGlobalMuon();
  bool isTrk = muon->isTrackerMuon();

  return ( isPF && (isGLB || isTrk) );
  //return ( isPF && isGLB );
}

bool MuonTrackProducer::isLooseMod(edm::Event& iEvent, reco::MuonCollection::const_iterator muon)
{
  //bool isPF = muon->isPFMuon();
  bool isGLB = muon->isGlobalMuon();
  bool isTrk = muon->isTrackerMuon();
  bool matchedSt = muon->numberOfMatchedStations() >= 1;
    
  return ( (isGLB || isTrk) && matchedSt );
}

bool MuonTrackProducer::isSoft(edm::Event& iEvent, reco::MuonCollection::const_iterator muon, bool useIPxy, bool useIPz)
{
  bool result = false;

  if (muon->muonBestTrack().isNonnull() && muon->innerTrack().isNonnull()){

 	edm::Handle<reco::VertexCollection> vertexHandle;
  	iEvent.getByLabel(vxtTag,vertexHandle);
  	const reco::VertexCollection* vertices = vertexHandle.product();

	bool isGood = muon::isGoodMuon((*muon), muon::TMOneStationTight); 
	bool trkLayMeas = muon->innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5; 
	bool pxlLayMeas = muon->innerTrack()->hitPattern().pixelLayersWithMeasurement() > 0; 
	bool quality = muon->innerTrack()->quality(reco::Track::highPurity);
	bool ipxy = false;
	bool ipz = false;
	if(vertices->size()!=0 && useIPxy) ipxy = fabs(muon->muonBestTrack()->dxy((*vertices)[0].position())) < 0.2;
	else ipxy = true;
 	if(vertices->size()!=0 && useIPz) ipz = fabs(muon->muonBestTrack()->dz((*vertices)[0].position())) < 0.5;
	else ipz = true;
	if(isGood && trkLayMeas && pxlLayMeas && quality && ipxy && ipz) result = true;

  }

  return result;
}

bool MuonTrackProducer::isTight(edm::Event& iEvent, reco::MuonCollection::const_iterator muon, bool useIPxy, bool useIPz)
{
  bool result = false;

  if (muon->muonBestTrack().isNonnull() && muon->innerTrack().isNonnull() && muon->globalTrack().isNonnull()){

	std::vector<double> vtxCoord = findSimVtx(iEvent);
        GlobalPoint point(vtxCoord[1],vtxCoord[2],vtxCoord[3]);
        GlobalPoint pointDY(vtxCoord[4],vtxCoord[5],vtxCoord[6]);

	//double muonX = muon->vx();
	//double muonY = muon->vy();
	//double muonZ = muon->vz();

	double muonZ = pointDY.z();
		
 	edm::Handle<reco::VertexCollection> vertexHandle;
  	iEvent.getByLabel(vxtTag,vertexHandle);
  	const reco::VertexCollection* vertices = vertexHandle.product();

	double distInit = 24;
	int indexFinal = 0;
	for(int i = 0; i < (int)vertices->size(); i++){

		//double vtxX = (*vertices)[i].x();
		//double vtxY = (*vertices)[i].y();
		double vtxZ = (*vertices)[i].z();

		double dist = fabs(muonZ - vtxZ);
		//std::cout<<"dist "<<dist<<std::endl;
		if(dist < distInit){
				
			distInit = dist;
			indexFinal = i;

		}

	}
	//std::cout<<distInit<<" "<<indexFinal<<std::endl;

	double ipxySim = 999;
	double ipzSim = 999;
	
	if(vtxCoord[0] > 1.5 && vtxCoord[0] < 3.5){

        	ipxySim = fabs(muon->muonBestTrack()->dxy(math::XYZPoint(point.x(),point.y(),point.z())));
        	ipzSim = fabs(muon->muonBestTrack()->dz(math::XYZPoint(point.x(),point.y(),point.z())));
	}
	else if(vtxCoord[0] > 0.5 && vtxCoord[0] < 1.5){

		ipxySim = fabs(muon->muonBestTrack()->dxy(math::XYZPoint(pointDY.x(),pointDY.y(),pointDY.z())));
        	ipzSim = fabs(muon->muonBestTrack()->dz(math::XYZPoint(pointDY.x(),pointDY.y(),pointDY.z())));

	}
	bool ipxySimBool = ipxySim < 0.2;
	bool ipzSimBool = ipzSim < 0.5;
        //std::cout<<"vx: "<<point.x()<<" vy: "<<point.y()<<" vz: "<<point.z()<<" |Dxy|: "<<ipxySim<<" "<<ipxySimBool<<" |Dz|: "<<ipzSim<<" "<<ipzSimBool<<std::endl;
        //std::cout<<"vx: "<<pointDY.x()<<" vy: "<<pointDY.y()<<" vz: "<<pointDY.z()<<" |Dxy|: "<<ipxySim<<" "<<ipxySimBool<<" |Dz|: "<<ipzSim<<" "<<ipzSimBool<<std::endl;

	bool trkLayMeas = muon->innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5; 
	bool isGlb = muon->isGlobalMuon(); 
	bool isPF = muon->isPFMuon(); 
	bool chi2 = muon->globalTrack()->normalizedChi2() < 10.; 
	bool validHits = muon->globalTrack()->hitPattern().numberOfValidMuonHits() > 0; 
	bool matchedSt = muon->numberOfMatchedStations() > 1;

	bool ipxy = false;
	bool ipz = false;
	if(vertices->size() !=0 && useIPxy == true){
	//if(useIPxy){

		if(vtxCoord[0] > 0.5 && vtxCoord[0] < 1.5) ipxy = fabs(muon->muonBestTrack()->dxy((*vertices)[indexFinal].position())) < 0.2;
		else if(vtxCoord[0] > 2.5 && vtxCoord[0] < 3.5) ipxy = fabs(muon->muonBestTrack()->dxy((*vertices)[0].position())) < 0.2;
		else ipxy = ipxySimBool;
		//ipxy = ipxySimBool;

	}
	else if(vertices->size() == 0 && useIPxy == true) ipxy = false;
	else if(useIPxy == false) ipxy = true;

 	if(vertices->size() !=0 && useIPz == true){
 	//if(useIPz){

		if(vtxCoord[0] > 0.5 && vtxCoord[0] < 1.5) ipz = fabs(muon->muonBestTrack()->dz((*vertices)[indexFinal].position())) < 0.5;
		else if(vtxCoord[0] > 2.5 && vtxCoord[0] < 3.5) ipz = fabs(muon->muonBestTrack()->dz((*vertices)[0].position())) < 0.5;	
		else ipz = ipzSimBool;
		//ipz = ipzSimBool;

        	//std::cout<<"vx: "<<pointDY.x()<<" vy: "<<pointDY.y()<<" vz: "<<pointDY.z()<<" |Dz|: "<<fabs(muon->muonBestTrack()->dz((*vertices)[indexFinal].position()))<<" "<<ipz<<std::endl;

	}
	else if(vertices->size() == 0 && useIPz == true) ipz = false;
	else if(useIPz == false) ipz = true;
	//bool validPxlHit = muon->innerTrack()->hitPattern().numberOfValidPixelHits() > 0;
	bool validPxlHit = muon->innerTrack()->hitPattern().pixelLayersWithMeasurement(3,2) > 0;
	//bool validPxlHit = muon->innerTrack()->hitPattern().pixelLayersWithMeasurement(4,3) > 0;

	if(trkLayMeas && isGlb && isPF && chi2 && validHits && matchedSt && ipxy && ipz && validPxlHit) result = true;

  }

  return result;
}

bool MuonTrackProducer::isTightBS(edm::Event& iEvent, reco::MuonCollection::const_iterator muon)
{
  bool result = false;

  if (muon->muonBestTrack().isNonnull() && muon->innerTrack().isNonnull() && muon->globalTrack().isNonnull()){
		
   	edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
   	iEvent.getByLabel("offlineBeamSpot", recoBeamSpotHandle);
   	reco::BeamSpot vertexBeamSpot= *recoBeamSpotHandle;
        //math::XYZPoint bs(vertexBeamSpot.x(), vertexBeamSpot.y(), vertexBeamSpot.z());

	bool trkLayMeas = muon->innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5; 
	bool isGlb = muon->isGlobalMuon(); 
	bool isPF = muon->isPFMuon(); 
	bool chi2 = muon->globalTrack()->normalizedChi2() < 10.; 
	bool validHits = muon->globalTrack()->hitPattern().numberOfValidMuonHits() > 0; 
	bool matchedSt = muon->numberOfMatchedStations() > 1;

	bool ipxy = false;
	ipxy = fabs(muon->muonBestTrack()->dxy(vertexBeamSpot.position())) < 0.2;

	//bool validPxlHit = muon->innerTrack()->hitPattern().numberOfValidPixelHits() > 0;
	bool validPxlHit = muon->innerTrack()->hitPattern().pixelLayersWithMeasurement(3,2) > 0;
	//bool validPxlHit = muon->innerTrack()->hitPattern().pixelLayersWithMeasurement(4,3) > 0;

	if(trkLayMeas && isGlb && isPF && chi2 && validHits && matchedSt && ipxy && validPxlHit) result = true;

  }

  return result;
}

bool MuonTrackProducer::isTightMod(edm::Event& iEvent, reco::MuonCollection::const_iterator muon, bool useIPxy, bool useIPz)
{
  bool result = false;

  if (muon->muonBestTrack().isNonnull() && muon->innerTrack().isNonnull() && muon->globalTrack().isNonnull()){

	std::vector<double> vtxCoord = findSimVtx(iEvent);
        GlobalPoint point(vtxCoord[1],vtxCoord[2],vtxCoord[3]);
        GlobalPoint pointDY(vtxCoord[4],vtxCoord[5],vtxCoord[6]);

	//double muonX = muon->vx();
	//double muonY = muon->vy();
	//double muonZ = muon->vz();

	double muonZ = pointDY.z();
		
 	edm::Handle<reco::VertexCollection> vertexHandle;
  	iEvent.getByLabel(vxtTag,vertexHandle);
  	const reco::VertexCollection* vertices = vertexHandle.product();

	double distInit = 24;
	int indexFinal = 0;
	for(int i = 0; i < (int)vertices->size(); i++){

		//double vtxX = (*vertices)[i].x();
		//double vtxY = (*vertices)[i].y();
		double vtxZ = (*vertices)[i].z();

		double dist = fabs(muonZ - vtxZ);
		//std::cout<<"dist "<<dist<<std::endl;
		if(dist < distInit){
				
			distInit = dist;
			indexFinal = i;

		}

	}
	//std::cout<<distInit<<" "<<indexFinal<<std::endl;

	double ipxySim = 999;
	double ipzSim = 999;
	
	if(vtxCoord[0] > 1.5 && vtxCoord[0] < 3.5){

        	ipxySim = fabs(muon->muonBestTrack()->dxy(math::XYZPoint(point.x(),point.y(),point.z())));
        	ipzSim = fabs(muon->muonBestTrack()->dz(math::XYZPoint(point.x(),point.y(),point.z())));
	}
	else if(vtxCoord[0] > 0.5 && vtxCoord[0] < 1.5){

		ipxySim = fabs(muon->muonBestTrack()->dxy(math::XYZPoint(pointDY.x(),pointDY.y(),pointDY.z())));
        	ipzSim = fabs(muon->muonBestTrack()->dz(math::XYZPoint(pointDY.x(),pointDY.y(),pointDY.z())));

	}
	bool ipxySimBool = ipxySim < 0.2;
	bool ipzSimBool = ipzSim < 0.5;
        //std::cout<<"vx: "<<point.x()<<" vy: "<<point.y()<<" vz: "<<point.z()<<" |Dxy|: "<<ipxySim<<" "<<ipxySimBool<<" |Dz|: "<<ipzSim<<" "<<ipzSimBool<<std::endl;
        //std::cout<<"vx: "<<pointDY.x()<<" vy: "<<pointDY.y()<<" vz: "<<pointDY.z()<<" |Dxy|: "<<ipxySim<<" "<<ipxySimBool<<" |Dz|: "<<ipzSim<<" "<<ipzSimBool<<std::endl;

	bool trkLayMeas = muon->innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5; 
	bool isGlb = muon->isGlobalMuon(); 
	//bool isPF = muon->isPFMuon();
	bool chi2 = muon->globalTrack()->normalizedChi2() < 10.; 
	bool validHits = muon->globalTrack()->hitPattern().numberOfValidMuonHits() > 0; 
	bool matchedSt = muon->numberOfMatchedStations() > 1;

	bool ipxy = false;
	bool ipz = false;
	if(vertices->size() !=0 && useIPxy == true){
	//if(useIPxy){

		if(vtxCoord[0] > 0.5 && vtxCoord[0] < 1.5) ipxy = fabs(muon->muonBestTrack()->dxy((*vertices)[indexFinal].position())) < 0.2;
		else if(vtxCoord[0] > 2.5 && vtxCoord[0] < 3.5) ipxy = fabs(muon->muonBestTrack()->dxy((*vertices)[0].position())) < 0.2;
		else ipxy = ipxySimBool;
		//ipxy = ipxySimBool;

	}
	else if(vertices->size() == 0 && useIPxy == true) ipxy = false;
	else if(useIPxy == false) ipxy = true;

 	if(vertices->size() !=0 && useIPz == true){
 	//if(useIPz){

		if(vtxCoord[0] > 0.5 && vtxCoord[0] < 1.5) ipz = fabs(muon->muonBestTrack()->dz((*vertices)[indexFinal].position())) < 0.5;
		else if(vtxCoord[0] > 2.5 && vtxCoord[0] < 3.5) ipz = fabs(muon->muonBestTrack()->dz((*vertices)[0].position())) < 0.5;	
		else ipz = ipzSimBool;
		//ipz = ipzSimBool;

        	//std::cout<<"vx: "<<pointDY.x()<<" vy: "<<pointDY.y()<<" vz: "<<pointDY.z()<<" |Dz|: "<<fabs(muon->muonBestTrack()->dz((*vertices)[indexFinal].position()))<<" "<<ipz<<std::endl;

	}
	else if(vertices->size() == 0 && useIPz == true) ipz = false;
	else if(useIPz == false) ipz = true;
	//bool validPxlHit = muon->innerTrack()->hitPattern().numberOfValidPixelHits() > 0;
	bool validPxlHit = muon->innerTrack()->hitPattern().pixelLayersWithMeasurement(3,2) > 0;
	//bool validPxlHit = muon->innerTrack()->hitPattern().pixelLayersWithMeasurement(4,3) > 0;

	if(trkLayMeas && isGlb && chi2 && validHits && matchedSt && ipxy && ipz && validPxlHit) result = true;

  }

  return result;
}


bool MuonTrackProducer::isLoose2(edm::Event& iEvent, reco::MuonCollection::const_iterator muon, bool useIPxy, bool useIPz)
{
  bool result = false;

  if (muon->muonBestTrack().isNonnull() && muon->innerTrack().isNonnull() && muon->globalTrack().isNonnull()){

	std::vector<double> vtxCoord = findSimVtx(iEvent);
        GlobalPoint point(vtxCoord[1],vtxCoord[2],vtxCoord[3]);
        GlobalPoint pointDY(vtxCoord[4],vtxCoord[5],vtxCoord[6]);

	//double muonX = muon->vx();
	//double muonY = muon->vy();
	//double muonZ = muon->vz();

	double muonZ = pointDY.z();
		
 	edm::Handle<reco::VertexCollection> vertexHandle;
  	iEvent.getByLabel(vxtTag,vertexHandle);
  	const reco::VertexCollection* vertices = vertexHandle.product();

	double distInit = 24;
	int indexFinal = 0;
	for(int i = 0; i < (int)vertices->size(); i++){

		//double vtxX = (*vertices)[i].x();
		//double vtxY = (*vertices)[i].y();
		double vtxZ = (*vertices)[i].z();

		double dist = fabs(muonZ - vtxZ);
		//std::cout<<"dist "<<dist<<std::endl;
		if(dist < distInit){
				
			distInit = dist;
			indexFinal = i;

		}

	}
	//std::cout<<distInit<<" "<<indexFinal<<std::endl;

	double ipxySim = 999;
	double ipzSim = 999;
	
	if(vtxCoord[0] > 1.5 && vtxCoord[0] < 3.5){

        	ipxySim = fabs(muon->muonBestTrack()->dxy(math::XYZPoint(point.x(),point.y(),point.z())));
        	ipzSim = fabs(muon->muonBestTrack()->dz(math::XYZPoint(point.x(),point.y(),point.z())));
	}
	else if(vtxCoord[0] > 0.5 && vtxCoord[0] < 1.5){

		ipxySim = fabs(muon->muonBestTrack()->dxy(math::XYZPoint(pointDY.x(),pointDY.y(),pointDY.z())));
        	ipzSim = fabs(muon->muonBestTrack()->dz(math::XYZPoint(pointDY.x(),pointDY.y(),pointDY.z())));

	}
	bool ipxySimBool = ipxySim < 0.2;
	bool ipzSimBool = ipzSim < 0.5;
        //std::cout<<"vx: "<<point.x()<<" vy: "<<point.y()<<" vz: "<<point.z()<<" |Dxy|: "<<ipxySim<<" "<<ipxySimBool<<" |Dz|: "<<ipzSim<<" "<<ipzSimBool<<std::endl;
        //std::cout<<"vx: "<<pointDY.x()<<" vy: "<<pointDY.y()<<" vz: "<<pointDY.z()<<" |Dxy|: "<<ipxySim<<" "<<ipxySimBool<<" |Dz|: "<<ipzSim<<" "<<ipzSimBool<<std::endl;

	bool isGlb = muon->isGlobalMuon(); 
	bool isPF = muon->isPFMuon(); 
  	//bool isTrk = muon->isTrackerMuon();

	bool ipxy = false;
	bool ipz = false;
	if(vertices->size() !=0 && useIPxy == true){
	//if(useIPxy){

		if(vtxCoord[0] > 0.5 && vtxCoord[0] < 1.5) ipxy = fabs(muon->muonBestTrack()->dxy((*vertices)[indexFinal].position())) < 0.2;
		else if(vtxCoord[0] > 2.5 && vtxCoord[0] < 3.5) ipxy = fabs(muon->muonBestTrack()->dxy((*vertices)[0].position())) < 0.2;
		else ipxy = ipxySimBool;
		//ipxy = ipxySimBool;

	}
	else if(vertices->size() == 0 && useIPxy == true) ipxy = false;
	else if(useIPxy == false) ipxy = true;

 	if(vertices->size() !=0 && useIPz == true){
 	//if(useIPz){

		if(vtxCoord[0] > 0.5 && vtxCoord[0] < 1.5) ipz = fabs(muon->muonBestTrack()->dz((*vertices)[indexFinal].position())) < 0.5;
		else if(vtxCoord[0] > 2.5 && vtxCoord[0] < 3.5) ipz = fabs(muon->muonBestTrack()->dz((*vertices)[0].position())) < 0.5;	
		else ipz = ipzSimBool;
		//ipz = ipzSimBool;

        	//std::cout<<"vx: "<<pointDY.x()<<" vy: "<<pointDY.y()<<" vz: "<<pointDY.z()<<" |Dz|: "<<fabs(muon->muonBestTrack()->dz((*vertices)[indexFinal].position()))<<" "<<ipz<<std::endl;

	}
	else if(vertices->size() == 0 && useIPz == true) ipz = false;
	else if(useIPz == false) ipz = true;

	//if((isGlb || isTrk) && isPF && ipxy && ipz) result = true;
	if(isGlb && isPF && ipxy && ipz) result = true;

  }

  return result;
}

MuonTrackProducer::MuonTrackProducer(const edm::ParameterSet& parset) :
  muonsTag(parset.getParameter< edm::InputTag >("muonsTag")),
  vxtTag(parset.getParameter< edm::InputTag >("vxtTag")),
  useIPxy(parset.getUntrackedParameter< bool >("useIPxy", true)),
  useIPz(parset.getUntrackedParameter< bool >("useIPz", true)),
  inputDTRecSegment4DCollection_(parset.getParameter<edm::InputTag>("inputDTRecSegment4DCollection")),
  inputCSCSegmentCollection_(parset.getParameter<edm::InputTag>("inputCSCSegmentCollection")),
  selectionTags(parset.getParameter< std::vector<std::string> >("selectionTags")),
  trackType(parset.getParameter< std::string >("trackType")),
  parset_(parset)
{
  edm::LogVerbatim("MuonTrackProducer") << "constructing  MuonTrackProducer" << parset_.dump();
  produces<reco::TrackCollection>();
  produces<reco::TrackExtraCollection>();
  produces<TrackingRecHitCollection>();
}

MuonTrackProducer::~MuonTrackProducer() {
}

void MuonTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
  iEvent.getByLabel(muonsTag,muonCollectionH);
  iEvent.getByLabel(inputDTRecSegment4DCollection_, dtSegmentCollectionH_);
  iEvent.getByLabel(inputCSCSegmentCollection_, cscSegmentCollectionH_);
  
  std::auto_ptr<reco::TrackCollection> selectedTracks(new reco::TrackCollection);
  std::auto_ptr<reco::TrackExtraCollection> selectedTrackExtras( new reco::TrackExtraCollection() );
  std::auto_ptr<TrackingRecHitCollection> selectedTrackHits( new TrackingRecHitCollection() );

  reco::TrackRefProd rTracks = iEvent.getRefBeforePut<reco::TrackCollection>();
  reco::TrackExtraRefProd rTrackExtras = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
  TrackingRecHitRefProd rHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();

  edm::Ref<reco::TrackExtraCollection>::key_type idx = 0;
  edm::Ref<reco::TrackExtraCollection>::key_type hidx = 0;

  edm::LogVerbatim("MuonTrackProducer") <<"\nThere are "<< dtSegmentCollectionH_->size()<<" DT segments.";
  unsigned int index_dt_segment = 0;
  for(DTRecSegment4DCollection::const_iterator segment = dtSegmentCollectionH_->begin();
      segment != dtSegmentCollectionH_->end(); ++segment , index_dt_segment++) {
    LocalPoint  segmentLocalPosition       = segment->localPosition();
    LocalVector segmentLocalDirection      = segment->localDirection();
    LocalError  segmentLocalPositionError  = segment->localPositionError();
    LocalError  segmentLocalDirectionError = segment->localDirectionError();
    DetId geoid = segment->geographicalId();
    DTChamberId dtdetid = DTChamberId(geoid);
    int wheel = dtdetid.wheel();
    int station = dtdetid.station();
    int sector = dtdetid.sector();
    
    float segmentX = segmentLocalPosition.x();
    float segmentY = segmentLocalPosition.y();
    float segmentdXdZ = segmentLocalDirection.x()/segmentLocalDirection.z();
    float segmentdYdZ = segmentLocalDirection.y()/segmentLocalDirection.z();
    float segmentXerr = sqrt(segmentLocalPositionError.xx());
    float segmentYerr = sqrt(segmentLocalPositionError.yy());
    float segmentdXdZerr = sqrt(segmentLocalDirectionError.xx());
    float segmentdYdZerr = sqrt(segmentLocalDirectionError.yy());

    edm::LogVerbatim("MuonTrackProducer") 
      <<"\nDT segment index :"<<index_dt_segment
      <<"\nchamber Wh:"<<wheel<<",St:"<<station<<",Se:"<<sector
      <<"\nLocal Position (X,Y)=("<<segmentX<<","<<segmentY<<") +/- ("<<segmentXerr<<","<<segmentYerr<<"), " 
      <<"Local Direction (dXdZ,dYdZ)=("<<segmentdXdZ<<","<<segmentdYdZ<<") +/- ("<<segmentdXdZerr<<","<<segmentdYdZerr<<")"; 
  }

  edm::LogVerbatim("MuonTrackProducer") <<"\nThere are "<< cscSegmentCollectionH_->size()<<" CSC segments.";
  unsigned int index_csc_segment = 0;
  for(CSCSegmentCollection::const_iterator segment = cscSegmentCollectionH_->begin();
      segment != cscSegmentCollectionH_->end(); ++segment , index_csc_segment++) {
    LocalPoint  segmentLocalPosition       = segment->localPosition();
    LocalVector segmentLocalDirection      = segment->localDirection();
    LocalError  segmentLocalPositionError  = segment->localPositionError();
    LocalError  segmentLocalDirectionError = segment->localDirectionError();

    DetId geoid = segment->geographicalId();
    CSCDetId cscdetid = CSCDetId(geoid);
    int endcap = cscdetid.endcap();
    int station = cscdetid.station();
    int ring = cscdetid.ring();
    int chamber = cscdetid.chamber(); 
    
    float segmentX = segmentLocalPosition.x();
    float segmentY = segmentLocalPosition.y();
    float segmentdXdZ = segmentLocalDirection.x()/segmentLocalDirection.z();
    float segmentdYdZ = segmentLocalDirection.y()/segmentLocalDirection.z();
    float segmentXerr = sqrt(segmentLocalPositionError.xx());
    float segmentYerr = sqrt(segmentLocalPositionError.yy());
    float segmentdXdZerr = sqrt(segmentLocalDirectionError.xx());
    float segmentdYdZerr = sqrt(segmentLocalDirectionError.yy());

    edm::LogVerbatim("MuonTrackProducer") 
      <<"\nCSC segment index :"<<index_csc_segment
      <<"\nchamber Endcap:"<<endcap<<",St:"<<station<<",Ri:"<<ring<<",Ch:"<<chamber
      <<"\nLocal Position (X,Y)=("<<segmentX<<","<<segmentY<<") +/- ("<<segmentXerr<<","<<segmentYerr<<"), " 
      <<"Local Direction (dXdZ,dYdZ)=("<<segmentdXdZ<<","<<segmentdYdZ<<") +/- ("<<segmentdXdZerr<<","<<segmentdYdZerr<<")"; 
  }

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
      bool looseMod = isLooseMod(iEvent, muon);
      bool loose2 = isLoose2(iEvent, muon, useIPxy, useIPz);
      //bool soft = isSoft(iEvent, muon, useIPxy, useIPz);
      bool tight = isTight(iEvent, muon, useIPxy, useIPz);
      bool tightBS = isTightBS(iEvent, muon);
      bool tightMod = isTightMod(iEvent, muon, useIPxy, useIPz);
      bool usingInner = false;
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
      else if (trackType == "bestMuon") {
	if (muon->muonBestTrack().isNonnull()) trackref = muon->muonBestTrack();
	else continue;
      }
      else if (trackType == "globalTrackLoose") {

	if(loose){

          	if( muon->globalTrack().isNonnull() )
                {
                   	trackref = muon->globalTrack();
	           	usingInner = false;
                }
          	else if ( muon->innerTrack().isNonnull() )
                {
                   	trackref = muon->innerTrack();
                   	usingInner = true;
                }
          	else continue;

	}

        //if (muon->muonBestTrack().isNonnull() && loose) trackref = muon->muonBestTrack();
        else continue;

      }
      else if (trackType == "globalTrackLooseMod") {

          if(looseMod){

          	if( muon->globalTrack().isNonnull() )
                {
                   	trackref = muon->globalTrack();
                    usingInner = false;
                }
          	else if ( muon->innerTrack().isNonnull() )
                {
                   	trackref = muon->innerTrack();
                   	usingInner = true;
                }
          	else continue;

          }

          //if (muon->muonBestTrack().isNonnull() && loose) trackref = muon->muonBestTrack();
          else continue;

      }
      else if (trackType == "globalTrackLoose2") {
        if (muon->globalTrack().isNonnull() && loose2) trackref = muon->globalTrack();
        else continue;
      }
      else if (trackType == "globalTrackTight") {
        if (muon->globalTrack().isNonnull() && tight) trackref = muon->globalTrack();
        else continue;
      }
      else if (trackType == "globalTrackTightMod") {
        if (muon->globalTrack().isNonnull() && tightMod) trackref = muon->globalTrack();
        else continue;
      }
      else if (trackType == "globalTrackTightBS") {
        if (muon->globalTrack().isNonnull() && tightBS) trackref = muon->globalTrack();
        else continue;
      }
      else if (trackType == "innerTrackPlusSegments") {
	if (muon->innerTrack().isNonnull()) trackref = muon->innerTrack();
	else continue;
      }

      const reco::Track* trk = &(*trackref);
      // pointer to old track:
      reco::Track* newTrk = new reco::Track(*trk);

      newTrk->setExtra( reco::TrackExtraRef( rTrackExtras, idx++ ) );
      PropagationDirection seedDir = trk->seedDirection();
      // new copy of track Extras
      reco::TrackExtra * newExtra = new reco::TrackExtra( trk->outerPosition(), trk->outerMomentum(), 
                                        trk->outerOk(), trk->innerPosition(), 
                                        trk->innerMomentum(), trk->innerOk(),
                                        trk->outerStateCovariance(), trk->outerDetId(),
                                        trk->innerStateCovariance(), trk->innerDetId() , seedDir ) ;

      // new copy of the silicon hits; add hit refs to Extra and hits to hit collection
      unsigned int index_hit = 0;
      
      //      edm::LogVerbatim("MuonTrackProducer")<<"\n printing initial hit_pattern";
      //      trk->hitPattern().print();
	
      for (trackingRecHit_iterator iHit = trk->recHitsBegin(); iHit != trk->recHitsEnd(); iHit++) {
        TrackingRecHit* hit = (*iHit)->clone();
	index_hit++;
        selectedTrackHits->push_back( hit );
        newExtra->add( TrackingRecHitRef( rHits, hidx++ ) );
      }

      if (trackType == "innerTrackPlusSegments" || usingInner) { 
	
	int wheel, station, sector;
	int endcap, /*station, */ ring, chamber;
	
	edm::LogVerbatim("MuonTrackProducer") <<"Number of chambers: "<<muon->matches().size()
					      <<", arbitrated: "<<muon->numberOfMatches(reco::Muon::SegmentAndTrackArbitration);
	unsigned int index_chamber = 0;
	
	for(std::vector<reco::MuonChamberMatch>::const_iterator chamberMatch = muon->matches().begin();
	    chamberMatch != muon->matches().end(); ++chamberMatch, index_chamber++) {
	  std::stringstream chamberStr;
	  chamberStr <<"\nchamber index: "<<index_chamber; 
	  
	  int subdet = chamberMatch->detector();
	  DetId did = chamberMatch->id;
	  
	  if (subdet == MuonSubdetId::DT) {
	    DTChamberId dtdetid = DTChamberId(did);
	    wheel = dtdetid.wheel();
	    station = dtdetid.station();
	    sector = dtdetid.sector();
	    chamberStr << ", DT chamber Wh:"<<wheel<<",St:"<<station<<",Se:"<<sector;
	  } 
	  else if (subdet == MuonSubdetId::CSC) {
	    CSCDetId cscdetid = CSCDetId(did);
	    endcap = cscdetid.endcap();
	    station = cscdetid.station();
	    ring = cscdetid.ring();
	    chamber = cscdetid.chamber();
	    chamberStr << ", CSC chamber End:"<<endcap<<",St:"<<station<<",Ri:"<<ring<<",Ch:"<<chamber;
	  }
	  
	  chamberStr << ", Number of segments: "<<chamberMatch->segmentMatches.size();
	  edm::LogVerbatim("MuonTrackProducer") << chamberStr.str();

	  unsigned int index_segment = 0;
	  
	  for(std::vector<reco::MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
	      segmentMatch != chamberMatch->segmentMatches.end(); ++segmentMatch, index_segment++) {
	    
	    float segmentX = segmentMatch->x;
	    float segmentY = segmentMatch->y ;
	    float segmentdXdZ = segmentMatch->dXdZ;
	    float segmentdYdZ = segmentMatch->dYdZ;
	    float segmentXerr = segmentMatch->xErr;
	    float segmentYerr = segmentMatch->yErr;
	    float segmentdXdZerr = segmentMatch->dXdZErr;
	    float segmentdYdZerr = segmentMatch->dYdZErr;
	    
	    CSCSegmentRef segmentCSC = segmentMatch->cscSegmentRef;
	    DTRecSegment4DRef segmentDT = segmentMatch->dtSegmentRef;
	    
	    bool segment_arbitrated_Ok = (segmentMatch->isMask(reco::MuonSegmentMatch::BestInChamberByDR) && 
					  segmentMatch->isMask(reco::MuonSegmentMatch::BelongsToTrackByDR));
	    
	    std::string ARBITRATED(" ***Arbitrated Off*** ");
	    if (segment_arbitrated_Ok) ARBITRATED = " ***ARBITRATED OK*** ";

	    if (subdet == MuonSubdetId::DT) {	      
	      edm::LogVerbatim("MuonTrackProducer")
		<<"\n\t segment index: "<<index_segment << ARBITRATED
		<<"\n\t  Local Position (X,Y)=("<<segmentX<<","<<segmentY<<") +/- ("<<segmentXerr<<","<<segmentYerr<<"), " 
		<<"\n\t  Local Direction (dXdZ,dYdZ)=("<<segmentdXdZ<<","<<segmentdYdZ<<") +/- ("<<segmentdXdZerr<<","<<segmentdYdZerr<<")"; 
	      
	      if (!segment_arbitrated_Ok) continue;
	      
	      if (segmentDT.get() != 0) {
		const DTRecSegment4D* segment = segmentDT.get();
		
		edm::LogVerbatim("MuonTrackProducer")<<"\t ===> MATCHING with DT segment with index = "<<segmentDT.key();
		
		if(segment->hasPhi()) {
		  const DTChamberRecSegment2D* phiSeg = segment->phiSegment();
		  std::vector<const TrackingRecHit*> phiHits = phiSeg->recHits();
		  for(std::vector<const TrackingRecHit*>::const_iterator ihit = phiHits.begin();
		      ihit != phiHits.end(); ++ihit) {
		    TrackingRecHit* seghit = (*ihit)->clone();
		    newTrk->setHitPattern( *seghit, index_hit);
		    //		    edm::LogVerbatim("MuonTrackProducer")<<"hit pattern for position "<<index_hit<<" set to:";
		    //		    newTrk->hitPattern().printHitPattern(index_hit, std::cout);
		    index_hit++;
		    selectedTrackHits->push_back( seghit );
		    newExtra->add( TrackingRecHitRef( rHits, hidx ++ ) );
		  }
		}
		
		if(segment->hasZed()) {
		  const DTSLRecSegment2D* zSeg = (*segment).zSegment();
		  std::vector<const TrackingRecHit*> zedHits = zSeg->recHits();
		  for(std::vector<const TrackingRecHit*>::const_iterator ihit = zedHits.begin();
		      ihit != zedHits.end(); ++ihit) {
		    TrackingRecHit* seghit = (*ihit)->clone();
		    newTrk->setHitPattern( *seghit, index_hit);
		    //		    edm::LogVerbatim("MuonTrackProducer")<<"hit pattern for position "<<index_hit<<" set to:";
		    //		    newTrk->hitPattern().printHitPattern(index_hit, std::cout);
		    index_hit++;
		    selectedTrackHits->push_back( seghit );
		    newExtra->add( TrackingRecHitRef( rHits, hidx ++ ) );
		  }
		}
	      } else edm::LogWarning("MuonTrackProducer")<<"\n***WARNING: UNMATCHED DT segment ! \n";
	    } // if (subdet == MuonSubdetId::DT)

	    else if (subdet == MuonSubdetId::CSC) {
	      edm::LogVerbatim("MuonTrackProducer")
		<<"\n\t segment index: "<<index_segment << ARBITRATED
		<<"\n\t  Local Position (X,Y)=("<<segmentX<<","<<segmentY<<") +/- ("<<segmentXerr<<","<<segmentYerr<<"), " 
		<<"\n\t  Local Direction (dXdZ,dYdZ)=("<<segmentdXdZ<<","<<segmentdYdZ<<") +/- ("<<segmentdXdZerr<<","<<segmentdYdZerr<<")"; 
	      
	      if (!segment_arbitrated_Ok) continue;
	      
	      if (segmentCSC.get() != 0) {
		const CSCSegment* segment = segmentCSC.get();
		
		edm::LogVerbatim("MuonTrackProducer")<<"\t ===> MATCHING with CSC segment with index = "<<segmentCSC.key();
		
		std::vector<const TrackingRecHit*> hits = segment->recHits();
		for(std::vector<const TrackingRecHit*>::const_iterator ihit = hits.begin();
		    ihit != hits.end(); ++ihit) {
		  TrackingRecHit* seghit = (*ihit)->clone();
		  newTrk->setHitPattern( *seghit, index_hit);
		  //		    edm::LogVerbatim("MuonTrackProducer")<<"hit pattern for position "<<index_hit<<" set to:";
		  //		    newTrk->hitPattern().printHitPattern(index_hit, std::cout);
		  index_hit++;
		  selectedTrackHits->push_back( seghit );
		  newExtra->add( TrackingRecHitRef( rHits, hidx ++ ) );		  
		}
	      } else edm::LogWarning("MuonTrackProducer")<<"\n***WARNING: UNMATCHED CSC segment ! \n";
	    }  //  else if (subdet == MuonSubdetId::CSC)

	  } // loop on vector<MuonSegmentMatch>	  
	} // loop on vector<MuonChamberMatch>	
      } // if (trackType == "innerTrackPlusSegments")
      
      //      edm::LogVerbatim("MuonTrackProducer")<<"\n printing final hit_pattern";
      //      newTrk->hitPattern().print();
      
      selectedTracks->push_back( *newTrk );
      selectedTrackExtras->push_back( *newExtra );

    } // if (isGoodResult)
  }  // loop on reco::MuonCollection
  
  iEvent.put(selectedTracks);
  iEvent.put(selectedTrackExtras);
  iEvent.put(selectedTrackHits);
}
