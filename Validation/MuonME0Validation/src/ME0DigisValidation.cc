#include "Validation/MuonME0Validation/interface/ME0DigisValidation.h"
#include <TMath.h>

ME0DigisValidation::ME0DigisValidation(const edm::ParameterSet& cfg):  ME0BaseValidation(cfg)
{
  InputTagToken_ = consumes<edm::PSimHitContainer>(cfg.getParameter<edm::InputTag>("simInputLabel"));
  InputTagToken_Digi = consumes<ME0DigiPreRecoCollection>(cfg.getParameter<edm::InputTag>("digiInputLabel"));
}

void ME0DigisValidation::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & Run, edm::EventSetup const & iSetup ) {

  LogDebug("MuonME0DigisValidation")<<"Info : Loading Geometry information\n";
  ibooker.setCurrentFolder("MuonME0DigisV/ME0DigisTask");

  unsigned int nregion  = 2;

  edm::LogInfo("MuonME0DigisValidation")<<"+++ Info : # of region : "<<nregion<<std::endl;

  LogDebug("MuonME0DigisValidation")<<"+++ Info : finish to get geometry information from ES.\n";
    
  me0_strip_dg_dx_local_tot_Muon = ibooker.book1D( "me0_strip_dg_dx_local_tot", "Local DeltaX; #Delta X_{local} [cm]; Entries", 50, -0.1, +0.1);
  me0_strip_dg_dy_local_tot_Muon = ibooker.book1D( "me0_strip_dg_dy_local_tot", "Local DeltaY; #Delta Y_{local} [cm]; Entries", 500, -10.0, +10.0);
  me0_strip_dg_dphi_global_tot_Muon = ibooker.book1D( "me0_strip_dg_dphi_global_tot", "Global DeltaPhi; #Delta #phi_{global} [rad]; Entries", 50, -0.01, +0.01);
    
  me0_strip_dg_dphi_vs_phi_global_tot_Muon = ibooker.book2D( "me0_strip_dg_dphi_vs_phi_global_tot", "Global DeltaPhi vs. Phi; #phi_{global} [rad]; #Delta #phi_{global} [rad]", 72,-M_PI,+M_PI,50,-0.01,+0.01);
    
  me0_strip_dg_den_eta_tot = ibooker.book1D( "me0_strip_dg_den_eta_tot", "Denominator; #eta; Entries", 12, 1.8, 3.0);
  me0_strip_dg_num_eta_tot = ibooker.book1D( "me0_strip_dg_num_eta_tot", "Numerator; #eta; Entries", 12, 1.8, 3.0);
    
  me0_strip_dg_bkg_eta_tot = ibooker.book1D( "me0_strip_dg_bkg_eta_tot", "Total neutron background; #eta; Entries", 12, 1.8, 3.0);
  me0_strip_dg_bkgElePos_eta = ibooker.book1D( "me0_strip_dg_bkgElePos_eta", "Neutron background: electrons+positrons; #eta; Entries", 12, 1.8, 3.0);
  me0_strip_dg_bkgNeutral_eta = ibooker.book1D( "me0_strip_dg_bkgNeutral_eta", "Neutron background: gammas+neutrons; #eta; Entries", 12, 1.8, 3.0);

  for( unsigned int region_num = 0 ; region_num < nregion ; region_num++ ) {
      me0_strip_dg_zr_tot[region_num] = BookHistZR(ibooker,"me0_strip_dg_tot","Digi",region_num);
      me0_strip_dg_zr_tot_Muon[region_num] = BookHistZR(ibooker,"me0_strip_dg_tot_Muon","Digi Muon",region_num);
      for( unsigned int layer_num = 0 ; layer_num < 6 ; layer_num++) {
          
          std::string hist_name_for_dx_local  = std::string("me0_strip_dg_dx_local")+regionLabel[region_num]+"_l"+layerLabel[layer_num];
          std::string hist_name_for_dy_local  = std::string("me0_strip_dg_dy_local")+regionLabel[region_num]+"_l"+layerLabel[layer_num];
          std::string hist_name_for_dphi_global  = std::string("me0_strip_dg_dphi_global")+regionLabel[region_num]+"_l"+layerLabel[layer_num];
          
          std::string hist_name_for_den_eta  = std::string("me0_strip_dg_den_eta")+regionLabel[region_num]+"_l"+layerLabel[layer_num];
          std::string hist_name_for_num_eta  = std::string("me0_strip_dg_num_eta")+regionLabel[region_num]+"_l"+layerLabel[layer_num];

          std::string hist_label_for_dx_local = "Local DeltaX: region"+regionLabel[region_num]+" layer "+layerLabel[layer_num]+" ; #Delta X_{local} [cm]; Entries";
          std::string hist_label_for_dy_local = "Local DeltaY: region"+regionLabel[region_num]+" layer "+layerLabel[layer_num]+" ; #Delta Y_{local} [cm]; Entries";
          std::string hist_label_for_dphi_global = "Global DeltaPhi: region"+regionLabel[region_num]+" layer "+layerLabel[layer_num]+" ; #Delta #phi_{global} [rad]; Entries";
          
          std::string hist_label_for_den_eta = "Denominator: region"+regionLabel[region_num]+" layer "+layerLabel[layer_num]+" ; #eta; Entries";
          std::string hist_label_for_num_eta = "Numerator: region"+regionLabel[region_num]+" layer "+layerLabel[layer_num]+" ; #eta; Entries";

          me0_strip_dg_xy[region_num][layer_num] = BookHistXY(ibooker,"me0_strip_dg","Digi",region_num,layer_num);
          me0_strip_dg_xy_Muon[region_num][layer_num] = BookHistXY(ibooker,"me0_strip_dg_Muon","Digi Muon",region_num,layer_num);
          
          me0_strip_dg_dx_local_Muon[region_num][layer_num] = ibooker.book1D( hist_name_for_dx_local.c_str(), hist_label_for_dx_local.c_str(), 50, -0.1, +0.1);
          me0_strip_dg_dy_local_Muon[region_num][layer_num] = ibooker.book1D( hist_name_for_dy_local.c_str(), hist_label_for_dy_local.c_str(), 500, -10.0, +10.0);
          me0_strip_dg_dphi_global_Muon[region_num][layer_num] = ibooker.book1D( hist_name_for_dphi_global.c_str(), hist_label_for_dphi_global.c_str(), 50, -0.01, +0.01);
          
          me0_strip_dg_den_eta[region_num][layer_num] = ibooker.book1D( hist_name_for_den_eta, hist_label_for_den_eta, 12, 1.8, 3.0);
          me0_strip_dg_num_eta[region_num][layer_num] = ibooker.book1D( hist_name_for_num_eta, hist_label_for_num_eta, 12, 1.8, 3.0);

      }
  }
}

ME0DigisValidation::~ME0DigisValidation() {
}


void ME0DigisValidation::analyze(const edm::Event& e,
                                     const edm::EventSetup& iSetup)
{
 edm::ESHandle<ME0Geometry> hGeom;
 iSetup.get<MuonGeometryRecord>().get(hGeom);
 const ME0Geometry* ME0Geometry_ =( &*hGeom);
  edm::Handle<edm::PSimHitContainer> ME0Hits;
  e.getByToken(InputTagToken_, ME0Hits);

  edm::Handle<ME0DigiPreRecoCollection> ME0Digis;
  e.getByToken(InputTagToken_Digi, ME0Digis);

  if (!ME0Hits.isValid() | !ME0Digis.isValid() ) {
    edm::LogError("ME0DigisValidation") << "Cannot get ME0Hits/ME0Digis by Token simInputTagToken";
    return ;
  }
    
  int count = 1;

  for (ME0DigiPreRecoCollection::DigiRangeIterator cItr=ME0Digis->begin(); cItr!=ME0Digis->end(); cItr++) {
    ME0DetId id = (*cItr).first;

    const GeomDet* gdet = ME0Geometry_->idToDet(id);
    if ( gdet == nullptr) {
      std::cout<<"Getting DetId failed. Discard this gem strip hit.Maybe it comes from unmatched geometry."<<std::endl;
      continue;
    }
    const BoundPlane & surface = gdet->surface();

    Short_t region = (Short_t) id.region();
    Short_t layer = (Short_t) id.layer();
    Short_t chamber = (Short_t) id.chamber();

    ME0DigiPreRecoCollection::const_iterator digiItr;
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
    {
        
      Short_t particleType = ((digiItr->pdgid())>>4);
      Short_t evtId = ((digiItr->pdgid())>>3)&1;
      Short_t bx = ((digiItr->pdgid())>>2)&1;
      Short_t procType = ((digiItr->pdgid())>>1)&1;
      Short_t prompt = (digiItr->pdgid())&1;
      LocalPoint lp(digiItr->x(), digiItr->y(), 0);

      GlobalPoint gp = surface.toGlobal(lp);

      Float_t g_r = (Float_t) gp.perp();
      Float_t g_x = (Float_t) gp.x();
      Float_t g_y = (Float_t) gp.y();
      Float_t g_z = (Float_t) gp.z();

      // fill hist
      int region_num = 0 ;
      if ( region == -1 ) region_num = 0 ;
      else if ( region == 1) region_num = 1;
      int layer_num = layer-1;

      if (abs(particleType) == 13 && evtId == 1 && bx == 1 && procType == 1 && prompt == 1) {
          
        me0_strip_dg_zr_tot_Muon[region_num]->Fill(g_z,g_r);
        me0_strip_dg_xy_Muon[region_num][layer_num]->Fill(g_x,g_y);
          
        for (auto hits=ME0Hits->begin(); hits!=ME0Hits->end(); hits++) {
            
            int pdgid = hits->particleType();
            int evtId = hits->eventId().event() == 0 ? 1 : 0;
            int bx = hits->eventId().bunchCrossing() == 0 ? 1 : 0;
            int procType = hits->processType() == 0 ? 1 : 0;
            
            if(!(abs(pdgid) == 13 && evtId == 1 && bx == 1 && procType == 1)) continue;
            
            const ME0DetId id(hits->detUnitId());
            Short_t region_sh = id.region();
            Short_t layer_sh = id.layer();
            Short_t chamber_sh = id.chamber();
            
            int region_sh_num = 0 ;
            if ( region_sh == -1 ) region_sh_num = 0 ;
            else if ( region_sh == 1) region_sh_num = 1;
            int layer_sh_num = layer_sh - 1;
            
            LocalPoint lp_sh = hits->localPosition();
            GlobalPoint gp_sh = surface.toGlobal(lp_sh);
            
            if(count == 1){
                
                me0_strip_dg_den_eta[region_sh_num][layer_sh_num]->Fill(fabs(gp_sh.eta()));
                me0_strip_dg_den_eta_tot->Fill(fabs(gp_sh.eta()));
                
            }
            
            if(isMatched(region, layer, chamber, region_sh, layer_sh, chamber_sh)){
        
                Float_t dx_loc = lp_sh.x()-lp.x();
                Float_t dy_loc = lp_sh.y()-lp.y();
                Float_t dphi_glob = gp_sh.phi()-gp.phi();
        
                me0_strip_dg_dx_local_Muon[region_num][layer_num]->Fill(dx_loc);
                me0_strip_dg_dy_local_Muon[region_num][layer_num]->Fill(dy_loc);
                me0_strip_dg_dphi_global_Muon[region_num][layer_num]->Fill(dphi_glob);
            
                me0_strip_dg_dx_local_tot_Muon->Fill(dx_loc);
                me0_strip_dg_dy_local_tot_Muon->Fill(dy_loc);
                me0_strip_dg_dphi_global_tot_Muon->Fill(dphi_glob);
            
                me0_strip_dg_dphi_vs_phi_global_tot_Muon->Fill(gp_sh.phi(),dphi_glob);
            
                me0_strip_dg_num_eta[region_num][layer_num]->Fill(fabs(gp_sh.eta()));
                me0_strip_dg_num_eta_tot->Fill(fabs(gp_sh.eta()));
            
            }
            
        }
    
        count++;
          
      }
      else {
        me0_strip_dg_zr_tot[region_num]->Fill(g_z,g_r);
        me0_strip_dg_xy[region_num][layer_num]->Fill(g_x,g_y);
      }
        
      if ((abs(particleType) == 11 || abs(particleType) == 22 || abs(particleType) == 2112) && evtId == 0 && procType == 0 && prompt == 0)
          me0_strip_dg_bkg_eta_tot->Fill(fabs(gp.eta()));
      if ((abs(particleType) == 11) && evtId == 0 && procType == 0 && prompt == 0)
          me0_strip_dg_bkgElePos_eta->Fill(fabs(gp.eta()));
      if ((abs(particleType) == 22 || abs(particleType) == 2112) && evtId == 0 && procType == 0 && prompt == 0)
          me0_strip_dg_bkgNeutral_eta->Fill(fabs(gp.eta()));
        
    }
  }

}


bool ME0DigisValidation::isMatched(const int region, const int layer, const int chamber, const int region_sh, const int layer_sh, const int chamber_sh)
{
    
    bool result = false;

    if(region == region_sh && layer == layer_sh && chamber == chamber_sh) result = true;

    return result;
        
}
