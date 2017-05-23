/** Derived from DTGeometryAnalyzer by Nicola Amapane
 *
 *  \author M. Maggi - INFN Bari
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include <memory>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <iomanip>
#include <set>

class ME0GeometryAnalyzer : public edm::one::EDAnalyzer<>
{
public: 
  ME0GeometryAnalyzer( const edm::ParameterSet& pset);

  ~ME0GeometryAnalyzer();

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
  
private:
  const std::string& myName() { return myName_;}

  const int dashedLineWidth_;
  const std::string dashedLine_;
  const std::string myName_;
  std::ofstream ofos;
};

using namespace std;

ME0GeometryAnalyzer::ME0GeometryAnalyzer( const edm::ParameterSet& /*iConfig*/ )
  : dashedLineWidth_(104), dashedLine_( string(dashedLineWidth_, '-') ), 
    myName_( "ME0GeometryAnalyzer" ) 
{ 
  ofos.open("ME0testOutput1EtaPart.out"); 
}

ME0GeometryAnalyzer::~ME0GeometryAnalyzer() 
{
  ofos.close();
}

void
ME0GeometryAnalyzer::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup )
{
  edm::ESHandle<ME0Geometry> pDD;
  iSetup.get<MuonGeometryRecord>().get(pDD);
	    
   for (auto ch : pDD->chambers()){
       
//    ME0DetId chId(ch->id());
//    ofos << "\"" << chId.rawId() << ":0.0\","<<endl;
       
       for (auto la : ch->layers()){
           ME0DetId laId(la->id());
           ofos << "\"" << laId.rawId() << ":0.0\","<<endl;

//    for (auto roll : layer->etaPartitions()){
//        
//      ME0DetId rId(roll->id());
////      ofos<<"\tME0EtaPartition , ME0DetId = " << rId.rawId() << ", " << rId << endl;
//      ofos << "\"" << rId.rawId() << ":0.0\","<<endl;;
//
//    }
           
       }
  }
  
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ME0GeometryAnalyzer);
