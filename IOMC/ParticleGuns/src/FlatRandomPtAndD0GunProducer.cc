/*
 *  \author Julia Yarba
 */

#include <ostream>

#include "IOMC/ParticleGuns/interface/FlatRandomPtAndD0GunProducer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "CLHEP/Random/RandFlat.h"

using namespace edm;
using namespace std;

FlatRandomPtAndD0GunProducer::FlatRandomPtAndD0GunProducer(const ParameterSet& pset) :
BaseFlatGunProducer(pset)
{
    
    
    ParameterSet defpset ;
    ParameterSet pgun_params =
        pset.getParameter<ParameterSet>("PGunParameters") ;
    
    fMinPt = pgun_params.getParameter<double>("MinPt");
    fMaxPt = pgun_params.getParameter<double>("MaxPt");
    LxyMin_ = pgun_params.getParameter<double>("LxyMin");
    LxyMax_ = pgun_params.getParameter<double>("LxyMax");
    LzMin_ = pgun_params.getParameter<double>("LzMin");
    LzMax_ = pgun_params.getParameter<double>("LzMax");
    dxyMin_ = pgun_params.getParameter<double>("dxyMin");
    dxyMax_ = pgun_params.getParameter<double>("dxyMax");
    
    produces<HepMCProduct>();
    produces<GenEventInfoProduct>();
}

FlatRandomPtAndD0GunProducer::~FlatRandomPtAndD0GunProducer()
{
    // no need to cleanup GenEvent memory - done in HepMCProduct
}

void FlatRandomPtAndD0GunProducer::produce(Event &e, const EventSetup& es)
{
    
    if ( fVerbosity > 0 )
    {
        cout << " FlatRandomPtAndD0GunProducer : Begin New Event Generation" << endl ;
    }
    // event loop (well, another step in it...)
    
    // no need to clean up GenEvent memory - done in HepMCProduct
    //
    
    // here re-create fEvt (memory)
    //
    fEvt = new HepMC::GenEvent() ;
    
    // now actualy, cook up the event from PDGTable and gun parameters
    //
    // 1st, primary vertex
    //
    double phi_vtx    = fRandomGenerator->fire(fMinPhi, fMaxPhi);
    double lenXY = 0;
    double lenZ = 0;
    double len_x = 0;
    double len_y = 0;
    double len_z = 0;
    HepMC::GenVertex* Vtx = 0;
    
    if( LxyMin_ == LxyMax_ ) {
        len_x = lenXY*cos(phi_vtx);
        len_y = lenXY*sin(phi_vtx);
    }
    else{
        lenXY = fRandomGenerator->fire(LxyMin_, LxyMax_)*10;
        len_x = lenXY*cos(phi_vtx);
        len_y = lenXY*sin(phi_vtx);
    }
    
    if( LzMin_ == LzMax_ ) {
        len_z = LzMin_;
    }
    else {
        lenZ = fRandomGenerator->fire(LzMin_, LzMax_)*10;
        len_z = lenZ;
    }
    //std::cout<<"LXY[mm]: "<<lenXY<<" LZ "<<lenZ<<" Lx: "<<len_x<<" Ly: "<<len_y<<" Lz: "<<len_z<<std::endl;
    Vtx = new HepMC::GenVertex(HepMC::FourVector(len_x,len_y,len_z));
    
    // loop over particles
    //
    int barcode = 1 ;
    for (unsigned int ip=0; ip<fPartIDs.size(); ++ip)
    {
        
        double pt     = 0;
        double eta    = 0;
        double phi    = 0;
        double dxySim = 999;
        
        do{
            
            pt     = fRandomGenerator->fire(fMinPt, fMaxPt) ;
            eta    = fRandomGenerator->fire(fMinEta, fMaxEta) ;
            phi    = fRandomGenerator->fire(fMinPhi, fMaxPhi) ;
            dxySim = (-Vtx->point3d().x()*sin(phi)+Vtx->point3d().y()*cos(phi));
            
        }while(!(fabs(dxySim) > dxyMin_*10 && fabs(dxySim) < dxyMax_*10));
        //std::cout<<" pT: "<<pt<<" eta: "<<eta<<" phi: "<<phi<<" dxy[mm]: "<<dxySim<<std::endl;
        
        int PartID = fPartIDs[ip] ;
        const HepPDT::ParticleData*
        PData = fPDGTable->particle(HepPDT::ParticleID(abs(PartID))) ;
        double mass   = PData->mass().value() ;
        double theta  = 2.*atan(exp(-eta)) ;
        double mom    = pt/sin(theta) ;
        double px     = pt*cos(phi) ;
        double py     = pt*sin(phi) ;
        double pz     = mom*cos(theta) ;
        double energy2= mom*mom + mass*mass ;
        double energy = sqrt(energy2) ;
        HepMC::FourVector p(px,py,pz,energy) ;
        HepMC::GenParticle* Part =
        new HepMC::GenParticle(p,PartID,1);
        Part->suggest_barcode( barcode ) ;
        barcode++ ;
        Vtx->add_particle_out(Part);
        
        if ( fAddAntiParticle )
        {
            HepMC::FourVector ap(-px,-py,-pz,energy) ;
            int APartID = -PartID ;
            if ( PartID == 22 || PartID == 23 )
            {
                APartID = PartID ;
            }
            HepMC::GenParticle* APart =
            new HepMC::GenParticle(ap,APartID,1);
            APart->suggest_barcode( barcode ) ;
            barcode++ ;
            Vtx->add_particle_out(APart) ;
        }
        
    }
    
    fEvt->add_vertex(Vtx) ;
    fEvt->set_event_number(e.id().event()) ;
    fEvt->set_signal_process_id(20) ;
    
    if ( fVerbosity > 0 )
    {
        fEvt->print() ;
    }
    
    auto_ptr<HepMCProduct> BProduct(new HepMCProduct()) ;
    BProduct->addHepMCData( fEvt );
    e.put(BProduct);
    
    auto_ptr<GenEventInfoProduct> genEventInfo(new GenEventInfoProduct(fEvt));
    e.put(genEventInfo);
    
    if ( true )
    {
        // for testing purpose only
        // fEvt->print() ; // prints empty info after it's made into edm::Event
        cout << " FlatRandomPtAndD0GunProducer : Event Generation Done " << endl;
    }
}
//#include "FWCore/Framework/interface/MakerMacros.h"
//DEFINE_FWK_MODULE(FlatRandomPtAndD0GunProducer);
