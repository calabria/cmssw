/** Derived from DTGeometryAnalyzer by Nicola Amapane
 *
 *  \author M. Maggi - INFN Bari
 */

#include <memory>
#include <fstream>
#include <FWCore/Framework/interface/Frameworkfwd.h>

#include <FWCore/Framework/interface/one/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

#include <string>
#include <cmath>
#include <vector>
#include <iomanip>
#include <set>

using namespace std;

class RPCGeometryAnalyzer : public edm::one::EDAnalyzer<> {

 public: 
  RPCGeometryAnalyzer( const edm::ParameterSet& pset);

  ~RPCGeometryAnalyzer();

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
 
  const std::string& myName() { return myName_;}

 private: 

  const int dashedLineWidth_;
  const std::string dashedLine_;
  const std::string myName_;
  std::ofstream ofos;
};

RPCGeometryAnalyzer::RPCGeometryAnalyzer( const edm::ParameterSet& /*iConfig*/ )
  : dashedLineWidth_(104), dashedLine_( std::string(dashedLineWidth_, '-') ), 
    myName_( "RPCGeometryAnalyzer" ) 
{ 
  ofos.open("RPCtestOutput.out"); 
}


RPCGeometryAnalyzer::~RPCGeometryAnalyzer() 
{
  ofos.close();
}

void
RPCGeometryAnalyzer::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup )
{
  edm::ESHandle<RPCGeometry> pDD;
  iSetup.get<MuonGeometryRecord>().get( pDD );

   for(TrackingGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){

//      //----------------------- RPCCHAMBER TEST -------------------------------------------------------

      if( dynamic_cast< const RPCChamber* >( *it ) != 0 ){
          
       const RPCChamber* ch = dynamic_cast< const RPCChamber* >( *it );

       std::vector< const RPCRoll*> rollsRaf = (ch->rolls());
       for(std::vector<const RPCRoll*>::iterator r = rollsRaf.begin(); r != rollsRaf.end(); ++r){

           if(((*r)->id().station() == 3 || (*r)->id().station() == 4) && (*r)->id().ring() == 1 && (*r)->id().region() != 0){
               
               std::cout<<"RPCDetId = "<<(*r)->id().rawId()<<std::endl;
               std::cout<<"Region = "<<(*r)->id().region()<<"  Ring = "<<(*r)->id().ring()<<"  Station = "<<(*r)->id().station()<<"  Sector = "<<(*r)->id().sector()<<"  Layer = "<<(*r)->id().layer()<<"  Subsector = "<<(*r)->id().subsector()<<"  Roll = "<<(*r)->id().roll()<<std::endl;
               ofos << "\"" << (*r)->id().rawId() << ":0.0\","<<std::endl;;
               
           }
       }
     }

    }
    std::cout <<std::endl;

}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(RPCGeometryAnalyzer);
