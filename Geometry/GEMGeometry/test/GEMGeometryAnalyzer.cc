
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

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include <memory>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <iomanip>
#include <set>

class GEMGeometryAnalyzer : public edm::one::EDAnalyzer<> {

public: 
  GEMGeometryAnalyzer( const edm::ParameterSet& pset);

  ~GEMGeometryAnalyzer();

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
GEMGeometryAnalyzer::GEMGeometryAnalyzer( const edm::ParameterSet& /*iConfig*/ )
  : dashedLineWidth_(104), dashedLine_( std::string(dashedLineWidth_, '-') ), 
    myName_( "GEMGeometryAnalyzer" ) 
{ 
  ofos.open("GEMtestOutput.out"); 
}


GEMGeometryAnalyzer::~GEMGeometryAnalyzer() 
{
  ofos.close();
}

void
GEMGeometryAnalyzer::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup )
{
  edm::ESHandle<GEMGeometry> pDD;
  iSetup.get<MuonGeometryRecord>().get(pDD);
  
  for (auto region : pDD->regions()) {
      
    for (auto station : region->stations()) {
        
      for (auto ring : station->rings()) {

          int i = 1;
          for (auto sch : ring->superChambers()) {
        
              GEMDetId schId(sch->id());
              int j = 1;
              
              for (auto ch : sch->chambers()){
          
                  GEMDetId chId(ch->id());
                  int k = 1;
                  auto& rolls(ch->etaPartitions());
	    
                  for (auto roll : rolls){
            
                      GEMDetId rId(roll->id());
                      if(rId.station() != 2) continue;
//                      ofos<<"Station: "<< rId.station() << " \"" << rId.rawId() << ":0.0\" " << rId << endl;
                      ofos<< "\"" << rId.rawId() << ":0.0\","<<endl;
            
                      ++k;
                  }
                  ++j;
              }
              ++i;
          }
        }
    }
  }

}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEMGeometryAnalyzer);
