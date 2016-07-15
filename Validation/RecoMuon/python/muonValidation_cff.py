import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *
# Configurations for MuonTrackValidators
import Validation.RecoMuon.MuonTrackValidator_cfi

#-----------------------------------------------------------------------------------------------------------------------

bestMuonLooseTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonLooseTrackVTrackAssoc.associatormap = 'tpToLooseMuonAssociation'
bestMuonLooseTrackVTrackAssoc.associators = ('MuonAssociationByHits',)
bestMuonLooseTrackVTrackAssoc.label = ('bestMuonLoose',)
bestMuonLooseTrackVTrackAssoc.usetracker = True
bestMuonLooseTrackVTrackAssoc.usemuon = True

bestMuonLooseTrackVTrackAssoc5 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonLooseTrackVTrackAssoc5.associatormap = 'tpToLoose5MuonAssociation'
bestMuonLooseTrackVTrackAssoc5.associators = ('MuonAssociationByHits',)
bestMuonLooseTrackVTrackAssoc5.label = ('bestMuonLoose5',)
bestMuonLooseTrackVTrackAssoc5.ptMinTP = 5.0
bestMuonLooseTrackVTrackAssoc5.usetracker = True
bestMuonLooseTrackVTrackAssoc5.usemuon = True

bestMuonLooseTrackVTrackAssoc10 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonLooseTrackVTrackAssoc10.associatormap = 'tpToLoose10MuonAssociation'
bestMuonLooseTrackVTrackAssoc10.associators = ('MuonAssociationByHits',)
bestMuonLooseTrackVTrackAssoc10.label = ('bestMuonLoose10',)
bestMuonLooseTrackVTrackAssoc10.ptMinTP = 10.0
bestMuonLooseTrackVTrackAssoc10.usetracker = True
bestMuonLooseTrackVTrackAssoc10.usemuon = True

bestMuonLooseTrackVTrackAssoc20 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonLooseTrackVTrackAssoc20.associatormap = 'tpToLoose20MuonAssociation'
bestMuonLooseTrackVTrackAssoc20.associators = ('MuonAssociationByHits',)
bestMuonLooseTrackVTrackAssoc20.label = ('bestMuonLoose20',)
bestMuonLooseTrackVTrackAssoc20.ptMinTP = 20.0
bestMuonLooseTrackVTrackAssoc20.usetracker = True
bestMuonLooseTrackVTrackAssoc20.usemuon = True

#-----------------------------------------------------------------------------------------------------------------------

bestMuonLoose2TrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonLoose2TrackVTrackAssoc.associatormap = 'tpToLoose2MuonAssociation'
bestMuonLoose2TrackVTrackAssoc.associators = ('MuonAssociationByHits',)
bestMuonLoose2TrackVTrackAssoc.label = ('bestMuonLoose2',)
bestMuonLoose2TrackVTrackAssoc.usetracker = True
bestMuonLoose2TrackVTrackAssoc.usemuon = True

bestMuonLoose2TrackVTrackAssoc5 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonLoose2TrackVTrackAssoc5.associatormap = 'tpToLoose52MuonAssociation'
bestMuonLoose2TrackVTrackAssoc5.associators = ('MuonAssociationByHits',)
bestMuonLoose2TrackVTrackAssoc5.label = ('bestMuonLoose52',)
bestMuonLoose2TrackVTrackAssoc5.ptMinTP = 5.0
bestMuonLoose2TrackVTrackAssoc5.usetracker = True
bestMuonLoose2TrackVTrackAssoc5.usemuon = True

bestMuonLoose2TrackVTrackAssoc10 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonLoose2TrackVTrackAssoc10.associatormap = 'tpToLoose102MuonAssociation'
bestMuonLoose2TrackVTrackAssoc10.associators = ('MuonAssociationByHits',)
bestMuonLoose2TrackVTrackAssoc10.label = ('bestMuonLoose102',)
bestMuonLoose2TrackVTrackAssoc10.ptMinTP = 10.0
bestMuonLoose2TrackVTrackAssoc10.usetracker = True
bestMuonLoose2TrackVTrackAssoc10.usemuon = True

bestMuonLoose2TrackVTrackAssoc20 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonLoose2TrackVTrackAssoc20.associatormap = 'tpToLoose202MuonAssociation'
bestMuonLoose2TrackVTrackAssoc20.associators = ('MuonAssociationByHits',)
bestMuonLoose2TrackVTrackAssoc20.label = ('bestMuonLoose202',)
bestMuonLoose2TrackVTrackAssoc20.ptMinTP = 20.0
bestMuonLoose2TrackVTrackAssoc20.usetracker = True
bestMuonLoose2TrackVTrackAssoc20.usemuon = True

#-----------------------------------------------------------------------------------------------------------------------

bestMuonTightTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonTightTrackVTrackAssoc.associatormap = 'tpToTightMuonAssociation'
bestMuonTightTrackVTrackAssoc.associators = ('MuonAssociationByHits',)
bestMuonTightTrackVTrackAssoc.label = ('bestMuonTight',)
bestMuonTightTrackVTrackAssoc.usetracker = True
bestMuonTightTrackVTrackAssoc.usemuon = True

bestMuonTightTrackVTrackAssoc5 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonTightTrackVTrackAssoc5.associatormap = 'tpToTight5MuonAssociation'
bestMuonTightTrackVTrackAssoc5.associators = ('MuonAssociationByHits',)
bestMuonTightTrackVTrackAssoc5.label = ('bestMuonTight5',)
bestMuonTightTrackVTrackAssoc5.ptMinTP = 5.0
bestMuonTightTrackVTrackAssoc5.usetracker = True
bestMuonTightTrackVTrackAssoc5.usemuon = True

bestMuonTightTrackVTrackAssoc10 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonTightTrackVTrackAssoc10.associatormap = 'tpToTight10MuonAssociation'
bestMuonTightTrackVTrackAssoc10.associators = ('MuonAssociationByHits',)
bestMuonTightTrackVTrackAssoc10.label = ('bestMuonTight10',)
bestMuonTightTrackVTrackAssoc10.ptMinTP = 10.0
bestMuonTightTrackVTrackAssoc10.usetracker = True
bestMuonTightTrackVTrackAssoc10.usemuon = True

bestMuonTightTrackVTrackAssoc20 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonTightTrackVTrackAssoc20.associatormap = 'tpToTight20MuonAssociation'
bestMuonTightTrackVTrackAssoc20.associators = ('MuonAssociationByHits',)
bestMuonTightTrackVTrackAssoc20.label = ('bestMuonTight20',)
bestMuonTightTrackVTrackAssoc20.ptMinTP = 20.0
bestMuonTightTrackVTrackAssoc20.usetracker = True
bestMuonTightTrackVTrackAssoc20.usemuon = True

#-----------------------------------------------------------------------------------------------------------------------

trkMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trkMuonTrackVTrackAssoc.associatormap = 'tpToTkmuTrackAssociation'
trkMuonTrackVTrackAssoc.associators = ('TrackAssociatorByHits',)
trkMuonTrackVTrackAssoc.label = ('generalTracks',)
#trkMuonTrackVTrackAssoc.label = ('probeTracks',)
trkMuonTrackVTrackAssoc.usetracker = True
trkMuonTrackVTrackAssoc.usemuon = False

trkCosmicMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trkCosmicMuonTrackVTrackAssoc.associatormap = 'tpToTkCosmicTrackAssociation'
trkCosmicMuonTrackVTrackAssoc.associators = ('TrackAssociatorByHits',)
trkCosmicMuonTrackVTrackAssoc.label = ('ctfWithMaterialTracksP5LHCNavigation',)
trkCosmicMuonTrackVTrackAssoc.usetracker = True
trkCosmicMuonTrackVTrackAssoc.usemuon = False

staMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staMuonTrackVTrackAssoc.associatormap = 'tpToStaTrackAssociation'
staMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staMuonTrackVTrackAssoc.label = ('standAloneMuons',)
staMuonTrackVTrackAssoc.usetracker = False
staMuonTrackVTrackAssoc.usemuon = True

staUpdMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVTrackAssoc.associatormap = 'tpToStaUpdTrackAssociation'
staUpdMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staUpdMuonTrackVTrackAssoc.label = ('standAloneMuons:UpdatedAtVtx',)
staUpdMuonTrackVTrackAssoc.usetracker = False
staUpdMuonTrackVTrackAssoc.usemuon = True

glbMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVTrackAssoc.associatormap = 'tpToGlbTrackAssociation'
glbMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
glbMuonTrackVTrackAssoc.label = ('globalMuons',)
glbMuonTrackVTrackAssoc.usetracker = True
glbMuonTrackVTrackAssoc.usemuon = True

staSETMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staSETMuonTrackVTrackAssoc.associatormap = 'tpToStaSETTrackAssociation'
staSETMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staSETMuonTrackVTrackAssoc.label = ('standAloneSETMuons',)
staSETMuonTrackVTrackAssoc.usetracker = False
staSETMuonTrackVTrackAssoc.usemuon = True

staSETUpdMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staSETUpdMuonTrackVTrackAssoc.associatormap = 'tpToStaSETUpdTrackAssociation'
staSETUpdMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staSETUpdMuonTrackVTrackAssoc.label = ('standAloneSETMuons:UpdatedAtVtx',)
staSETUpdMuonTrackVTrackAssoc.usetracker = False
staSETUpdMuonTrackVTrackAssoc.usemuon = True

glbSETMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbSETMuonTrackVTrackAssoc.associatormap = 'tpToGlbSETTrackAssociation'
glbSETMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
glbSETMuonTrackVTrackAssoc.label = ('globalSETMuons',)
glbSETMuonTrackVTrackAssoc.usetracker = True
glbSETMuonTrackVTrackAssoc.usemuon = True

tevMuonFirstTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonFirstTrackVTrackAssoc.associatormap = 'tpToTevFirstTrackAssociation'
tevMuonFirstTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
tevMuonFirstTrackVTrackAssoc.label = ('tevMuons:firstHit',)
tevMuonFirstTrackVTrackAssoc.usetracker = True
tevMuonFirstTrackVTrackAssoc.usemuon = True

tevMuonPickyTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonPickyTrackVTrackAssoc.associatormap = 'tpToTevPickyTrackAssociation'
tevMuonPickyTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
tevMuonPickyTrackVTrackAssoc.label = ('tevMuons:picky',)
tevMuonPickyTrackVTrackAssoc.usetracker = True
tevMuonPickyTrackVTrackAssoc.usemuon = True

tevMuonDytTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonDytTrackVTrackAssoc.associatormap = 'tpToTevDytTrackAssociation'
tevMuonDytTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
tevMuonDytTrackVTrackAssoc.label = ('tevMuons:dyt',)
tevMuonDytTrackVTrackAssoc.usetracker = True
tevMuonDytTrackVTrackAssoc.usemuon = True

staCosmicMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staCosmicMuonTrackVTrackAssoc.associatormap = 'tpToStaCosmicTrackAssociation'
staCosmicMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staCosmicMuonTrackVTrackAssoc.label = ('cosmicMuons',)
staCosmicMuonTrackVTrackAssoc.usetracker = False
staCosmicMuonTrackVTrackAssoc.usemuon = True

glbCosmicMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbCosmicMuonTrackVTrackAssoc.associatormap = 'tpToGlbCosmicTrackAssociation'
glbCosmicMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
glbCosmicMuonTrackVTrackAssoc.label = ('globalCosmicMuons',)
glbCosmicMuonTrackVTrackAssoc.usetracker = True
glbCosmicMuonTrackVTrackAssoc.usemuon = True

trkProbeTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
#trkMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trkProbeTrackVMuonAssoc.associatormap = 'tpToTkMuonAssociation' 
trkProbeTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
trkProbeTrackVMuonAssoc.label = ('generalTracks',)
#trkProbeTrackVMuonAssoc.label = ('probeTracks',)
trkProbeTrackVMuonAssoc.usetracker = True
trkProbeTrackVMuonAssoc.usemuon = False

#staSeedTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
#staSeedTrackVMuonAssoc.associatormap = 'tpToStaSeedAssociation'
#staSeedTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
#staSeedTrackVMuonAssoc.label = ('seedsOfSTAmuons',)
#staSeedTrackVMuonAssoc.usetracker = False
#staSeedTrackVMuonAssoc.usemuon = True

staMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staMuonTrackVMuonAssoc.associatormap = 'tpToStaMuonAssociation'
staMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staMuonTrackVMuonAssoc.label = ('standAloneMuons',)
staMuonTrackVMuonAssoc.usetracker = False
staMuonTrackVMuonAssoc.usemuon = True
staMuonTrackVMuonAssoc.tipTP = 300000.
staMuonTrackVMuonAssoc.lipTP = 300000.
staMuonTrackVMuonAssoc.vertexSrc = ""
#staMuonTrackVMuonAssoc.stableOnlyTP = False

staUpdMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVMuonAssoc.associatormap = 'tpToStaUpdMuonAssociation'
staUpdMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staUpdMuonTrackVMuonAssoc.label = ('standAloneMuons:UpdatedAtVtx',)
staUpdMuonTrackVMuonAssoc.usetracker = False
staUpdMuonTrackVMuonAssoc.usemuon = True
staUpdMuonTrackVMuonAssoc.tipTP = 300000.
staUpdMuonTrackVMuonAssoc.lipTP = 300000.
staUpdMuonTrackVMuonAssoc.vertexSrc = ""
#staUpdMuonTrackVMuonAssoc.stableOnlyTP = False

staUpdMuonTrackVMuonAssoc2St = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVMuonAssoc2St.associatormap = 'tpToStaUpdMuonAssociation2St'
staUpdMuonTrackVMuonAssoc2St.associators = ('MuonAssociationByHits',)
staUpdMuonTrackVMuonAssoc2St.label = ('extractedSTAMuons2Stat',)
staUpdMuonTrackVMuonAssoc2St.usetracker = False
staUpdMuonTrackVMuonAssoc2St.usemuon = True

staUpdMuonTrackVMuonAssoc2StTime = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVMuonAssoc2StTime.associatormap = 'tpToStaUpdMuonAssociation2StTime'
staUpdMuonTrackVMuonAssoc2StTime.associators = ('MuonAssociationByHits',)
staUpdMuonTrackVMuonAssoc2StTime.label = ('extractedSTAMuons2StatTiming',)
staUpdMuonTrackVMuonAssoc2StTime.usetracker = False
staUpdMuonTrackVMuonAssoc2StTime.usemuon = True

glbMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVMuonAssoc.associatormap = 'tpToGlbMuonAssociation'
glbMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
glbMuonTrackVMuonAssoc.label = ('extractedGlobalMuons',)
glbMuonTrackVMuonAssoc.usetracker = True
glbMuonTrackVMuonAssoc.usemuon = True

staRefitMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staRefitMuonTrackVMuonAssoc.associatormap = 'tpToStaRefitMuonAssociation'
staRefitMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staRefitMuonTrackVMuonAssoc.label = ('refittedStandAloneMuons',)
staRefitMuonTrackVMuonAssoc.usetracker = False
staRefitMuonTrackVMuonAssoc.usemuon = True
staRefitMuonTrackVMuonAssoc.tipTP = 300000.
staRefitMuonTrackVMuonAssoc.lipTP = 300000.
staRefitMuonTrackVMuonAssoc.vertexSrc = ""
#staRefitMuonTrackVMuonAssoc.stableOnlyTP = False

staRefitUpdMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staRefitUpdMuonTrackVMuonAssoc.associatormap = 'tpToStaRefitUpdMuonAssociation'
staRefitUpdMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staRefitUpdMuonTrackVMuonAssoc.label = ('refittedStandAloneMuons:UpdatedAtVtx',)
staRefitUpdMuonTrackVMuonAssoc.usetracker = False
staRefitUpdMuonTrackVMuonAssoc.usemuon = True
staRefitUpdMuonTrackVMuonAssoc.tipTP = 300000.
staRefitUpdMuonTrackVMuonAssoc.lipTP = 300000.
staRefitUpdMuonTrackVMuonAssoc.vertexSrc = ""
#staRefitUpdMuonTrackVMuonAssoc.stableOnlyTP = False

staSETMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staSETMuonTrackVMuonAssoc.associatormap = 'tpToStaSETMuonAssociation'
staSETMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staSETMuonTrackVMuonAssoc.label = ('standAloneSETMuons',)
staSETMuonTrackVMuonAssoc.usetracker = False
staSETMuonTrackVMuonAssoc.usemuon = True

staSETUpdMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staSETUpdMuonTrackVMuonAssoc.associatormap = 'tpToStaSETUpdMuonAssociation'
staSETUpdMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staSETUpdMuonTrackVMuonAssoc.label = ('standAloneSETMuons:UpdatedAtVtx',)
staSETUpdMuonTrackVMuonAssoc.usetracker = False
staSETUpdMuonTrackVMuonAssoc.usemuon = True

glbSETMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbSETMuonTrackVMuonAssoc.associatormap = 'tpToGlbSETMuonAssociation'
glbSETMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
glbSETMuonTrackVMuonAssoc.label = ('globalSETMuons',)
glbSETMuonTrackVMuonAssoc.usetracker = True
glbSETMuonTrackVMuonAssoc.usemuon = True

tevMuonFirstTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonFirstTrackVMuonAssoc.associatormap = 'tpToTevFirstMuonAssociation'
tevMuonFirstTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
tevMuonFirstTrackVMuonAssoc.label = ('tevMuons:firstHit',)
tevMuonFirstTrackVMuonAssoc.usetracker = True
tevMuonFirstTrackVMuonAssoc.usemuon = True

tevMuonPickyTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonPickyTrackVMuonAssoc.associatormap = 'tpToTevPickyMuonAssociation'
tevMuonPickyTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
tevMuonPickyTrackVMuonAssoc.label = ('tevMuons:picky',)
tevMuonPickyTrackVMuonAssoc.usetracker = True
tevMuonPickyTrackVMuonAssoc.usemuon = True

tevMuonDytTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonDytTrackVMuonAssoc.associatormap = 'tpToTevDytMuonAssociation'
tevMuonDytTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
tevMuonDytTrackVMuonAssoc.label = ('tevMuons:dyt',)
tevMuonDytTrackVMuonAssoc.usetracker = True
tevMuonDytTrackVMuonAssoc.usemuon = True

staCosmicMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staCosmicMuonTrackVMuonAssoc.associatormap = 'tpToStaCosmicMuonAssociation'
staCosmicMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staCosmicMuonTrackVMuonAssoc.label = ('cosmicMuons',)
staCosmicMuonTrackVMuonAssoc.usetracker = False
staCosmicMuonTrackVMuonAssoc.usemuon = True

glbCosmicMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbCosmicMuonTrackVMuonAssoc.associatormap = 'tpToGlbCosmicMuonAssociation'
glbCosmicMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
glbCosmicMuonTrackVMuonAssoc.label = ('globalCosmicMuons',)
glbCosmicMuonTrackVMuonAssoc.usetracker = True
glbCosmicMuonTrackVMuonAssoc.usemuon = True

### a few more validator modules usable also for Upgrade TP studies

# about STA
trkProbeTrackVSelMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trkProbeTrackVSelMuonAssoc.associatormap = 'tpToTkSelMuonAssociation'
trkProbeTrackVSelMuonAssoc.associators = ('MuonAssociationByHits',)
trkProbeTrackVSelMuonAssoc.label = ('probeTracks',)
trkProbeTrackVSelMuonAssoc.usetracker = True
trkProbeTrackVSelMuonAssoc.usemuon = False

staUpdMuonTrackVSelMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVSelMuonAssoc.associatormap = 'tpToStaUpdSelMuonAssociation'
staUpdMuonTrackVSelMuonAssoc.associators = ('MuonAssociationByHits',)
staUpdMuonTrackVSelMuonAssoc.label = ('standAloneMuons:UpdatedAtVtx',)
staUpdMuonTrackVSelMuonAssoc.usetracker = False
staUpdMuonTrackVSelMuonAssoc.usemuon = True

staUpdMuonTrackVSelMuonAssoc2St = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVSelMuonAssoc2St.associatormap = 'tpToStaUpdSelMuonAssociation2St'
staUpdMuonTrackVSelMuonAssoc2St.associators = ('MuonAssociationByHits',)
staUpdMuonTrackVSelMuonAssoc2St.label = ('extractedSTAMuons2Stat',)
staUpdMuonTrackVSelMuonAssoc2St.usetracker = False
staUpdMuonTrackVSelMuonAssoc2St.usemuon = True

staUpdMuonTrackVSelMuonAssoc2StTime = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVSelMuonAssoc2StTime.associatormap = 'tpToStaUpdSelMuonAssociation2StTime'
staUpdMuonTrackVSelMuonAssoc2StTime.associators = ('MuonAssociationByHits',)
staUpdMuonTrackVSelMuonAssoc2StTime.label = ('extractedSTAMuons2StatTiming',)
staUpdMuonTrackVSelMuonAssoc2StTime.usetracker = False
staUpdMuonTrackVSelMuonAssoc2StTime.usemuon = True

staUpdMuonTrackVSel2MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVSel2MuonAssoc.associatormap = 'tpToStaUpdSel2MuonAssociation'
staUpdMuonTrackVSel2MuonAssoc.associators = ('MuonAssociationByHits',)
staUpdMuonTrackVSel2MuonAssoc.label = ('standAloneMuons:UpdatedAtVtx',)
staUpdMuonTrackVSel2MuonAssoc.usetracker = False
staUpdMuonTrackVSel2MuonAssoc.usemuon = True

staUpdMuonTrackVSel2MuonAssoc2St = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVSel2MuonAssoc2St.associatormap = 'tpToStaUpdSel2MuonAssociation2St'
staUpdMuonTrackVSel2MuonAssoc2St.associators = ('MuonAssociationByHits',)
staUpdMuonTrackVSel2MuonAssoc2St.label = ('extractedSTAMuons2Stat',)
staUpdMuonTrackVSel2MuonAssoc2St.usetracker = False
staUpdMuonTrackVSel2MuonAssoc2St.usemuon = True

staUpdMuonTrackVSel2MuonAssoc2StTime = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVSel2MuonAssoc2StTime.associatormap = 'tpToStaUpdSel2MuonAssociation2StTime'
staUpdMuonTrackVSel2MuonAssoc2StTime.associators = ('MuonAssociationByHits',)
staUpdMuonTrackVSel2MuonAssoc2StTime.label = ('extractedSTAMuons2StatTiming',)
staUpdMuonTrackVSel2MuonAssoc2StTime.usetracker = False
staUpdMuonTrackVSel2MuonAssoc2StTime.usemuon = True
#staUpdMuonTrackVSel2MuonAssoc2StTime.ptMinTP = 5.0

staUpdMuonTrackVSel2MuonAssoc2StTime05 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVSel2MuonAssoc2StTime05.associatormap = 'tpToStaUpdSel2MuonAssociation2StTime'
staUpdMuonTrackVSel2MuonAssoc2StTime05.associators = ('MuonAssociationByHits',)
staUpdMuonTrackVSel2MuonAssoc2StTime05.label = ('extractedSTAMuons2StatTiming',)
staUpdMuonTrackVSel2MuonAssoc2StTime05.usetracker = False
staUpdMuonTrackVSel2MuonAssoc2StTime05.usemuon = True
staUpdMuonTrackVSel2MuonAssoc2StTime05.ptMinTP = 5.0
staUpdMuonTrackVSel2MuonAssoc2StTime05.dirName = 'Muons/RecoMuonV/MultiTrack/Cut5/'

staUpd10SelMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpd10SelMuonTrackVMuonAssoc.associatormap = 'tpToStaUpd10SelMuonAssociation'
staUpd10SelMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staUpd10SelMuonTrackVMuonAssoc.label = ('staMuonsPt10',)
staUpd10SelMuonTrackVMuonAssoc.usetracker = False
staUpd10SelMuonTrackVMuonAssoc.usemuon = True

staUpd20SelMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpd20SelMuonTrackVMuonAssoc.associatormap = 'tpToStaUpd20SelMuonAssociation'
staUpd20SelMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staUpd20SelMuonTrackVMuonAssoc.label = ('staMuonsPt20',)
staUpd20SelMuonTrackVMuonAssoc.usetracker = False
staUpd20SelMuonTrackVMuonAssoc.usemuon = True

# about GLB
glbMuonTrackVSelMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVSelMuonAssoc.associatormap = 'tpToGlbSelMuonAssociation'
glbMuonTrackVSelMuonAssoc.associators = ('MuonAssociationByHits',)
glbMuonTrackVSelMuonAssoc.label = ('extractedGlobalMuons',)
glbMuonTrackVSelMuonAssoc.usetracker = True
glbMuonTrackVSelMuonAssoc.usemuon = True

glbMuonTrackVSel2MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVSel2MuonAssoc.associatormap = 'tpToGlbSel2MuonAssociation'
glbMuonTrackVSel2MuonAssoc.associators = ('MuonAssociationByHits',)
glbMuonTrackVSel2MuonAssoc.label = ('extractedGlobalMuons',)
glbMuonTrackVSel2MuonAssoc.usetracker = True
glbMuonTrackVSel2MuonAssoc.usemuon = True

glbMuonTrackVSel3MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVSel3MuonAssoc.associatormap = 'tpToGlbSel3MuonAssociation'
glbMuonTrackVSel3MuonAssoc.associators = ('MuonAssociationByHits',)
glbMuonTrackVSel3MuonAssoc.label = ('extractedGlobalMuons',)
glbMuonTrackVSel3MuonAssoc.usetracker = True
glbMuonTrackVSel3MuonAssoc.usemuon = True

glbMuonTrackVSel4MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVSel4MuonAssoc.associatormap = 'tpToGlbSel4MuonAssociation'
glbMuonTrackVSel4MuonAssoc.associators = ('MuonAssociationByHits',)
glbMuonTrackVSel4MuonAssoc.label = ('extractedGlobalMuons',)
glbMuonTrackVSel4MuonAssoc.usetracker = True
glbMuonTrackVSel4MuonAssoc.usemuon = True

##################################################################################

looseMuonTrackVSelMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
looseMuonTrackVSelMuonAssoc.associatormap = 'tpToLooseSelMuonAssociation'
looseMuonTrackVSelMuonAssoc.associators = ('MuonAssociationByHits',)
looseMuonTrackVSelMuonAssoc.label = ('bestMuonLoose',)
looseMuonTrackVSelMuonAssoc.usetracker = True
looseMuonTrackVSelMuonAssoc.usemuon = True

looseMuonTrackVSel2MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
looseMuonTrackVSel2MuonAssoc.associatormap = 'tpToLooseSel2MuonAssociation'
looseMuonTrackVSel2MuonAssoc.associators = ('MuonAssociationByHits',)
looseMuonTrackVSel2MuonAssoc.label = ('bestMuonLoose',)
looseMuonTrackVSel2MuonAssoc.usetracker = True
looseMuonTrackVSel2MuonAssoc.usemuon = True

looseMuonTrackVSel0MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
looseMuonTrackVSel0MuonAssoc.associatormap = 'tpToLooseSel0MuonAssociation'
looseMuonTrackVSel0MuonAssoc.associators = ('MuonAssociationByHits',)
looseMuonTrackVSel0MuonAssoc.label = ('bestMuonLoose',)
looseMuonTrackVSel0MuonAssoc.usetracker = True
looseMuonTrackVSel0MuonAssoc.usemuon = True
looseMuonTrackVSel0MuonAssoc.useMCTruth = False

looseMuonTrackVSelUncMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
looseMuonTrackVSelUncMuonAssoc.associatormap = 'tpToLooseSelUncMuonAssociation'
looseMuonTrackVSelUncMuonAssoc.associators = ('MuonAssociationByHits',)
looseMuonTrackVSelUncMuonAssoc.label = ('bestMuonLoose',)
looseMuonTrackVSelUncMuonAssoc.usetracker = False
looseMuonTrackVSelUncMuonAssoc.usemuon = True
looseMuonTrackVSelUncMuonAssoc.useMCTruth = False

looseMuonTrackVSel05SimMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
looseMuonTrackVSel05SimMuonAssoc.associatormap = 'tpToLooseSel0MuonAssociation'
looseMuonTrackVSel05SimMuonAssoc.associators = ('MuonAssociationByHits',)
looseMuonTrackVSel05SimMuonAssoc.label = ('bestMuonLoose',)
looseMuonTrackVSel05SimMuonAssoc.ptMinTP = 5.0
looseMuonTrackVSel05SimMuonAssoc.usetracker = True
looseMuonTrackVSel05SimMuonAssoc.usemuon = True
looseMuonTrackVSel05SimMuonAssoc.useMCTruth = False
looseMuonTrackVSel05SimMuonAssoc.dirName = 'Muons/RecoMuonV/MultiTrack/Cut5/'

looseMuonTrackVSel05MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
looseMuonTrackVSel05MuonAssoc.associatormap = 'tpToLooseSel05MuonAssociation'
looseMuonTrackVSel05MuonAssoc.associators = ('MuonAssociationByHits',)
looseMuonTrackVSel05MuonAssoc.label = ('bestMuonLoose5',)
looseMuonTrackVSel05MuonAssoc.ptMinTP = 5.0
looseMuonTrackVSel05MuonAssoc.usetracker = True
looseMuonTrackVSel05MuonAssoc.usemuon = True
looseMuonTrackVSel05MuonAssoc.useMCTruth = False

#--------------------------------------------------------------------------------------------------------------------

looseModMuonTrackVSel0MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
looseModMuonTrackVSel0MuonAssoc.associatormap = 'tpToLooseModSel0MuonAssociation'
looseModMuonTrackVSel0MuonAssoc.associators = ('MuonAssociationByHits',)
looseModMuonTrackVSel0MuonAssoc.label = ('bestMuonLooseMod',)
looseModMuonTrackVSel0MuonAssoc.usetracker = True
looseModMuonTrackVSel0MuonAssoc.usemuon = True
looseModMuonTrackVSel0MuonAssoc.useMCTruth = False

looseModMuonTrackVSelUncMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
looseModMuonTrackVSelUncMuonAssoc.associatormap = 'tpToLooseModSelUncMuonAssociation'
looseModMuonTrackVSelUncMuonAssoc.associators = ('MuonAssociationByHits',)
looseModMuonTrackVSelUncMuonAssoc.label = ('bestMuonLooseMod',)
looseModMuonTrackVSelUncMuonAssoc.usetracker = False
looseModMuonTrackVSelUncMuonAssoc.usemuon = True
looseModMuonTrackVSelUncMuonAssoc.useMCTruth = False

looseModMuonTrackVSel05SimMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
looseModMuonTrackVSel05SimMuonAssoc.associatormap = 'tpToLooseModSel0MuonAssociation'
looseModMuonTrackVSel05SimMuonAssoc.associators = ('MuonAssociationByHits',)
looseModMuonTrackVSel05SimMuonAssoc.label = ('bestMuonLooseMod',)
looseModMuonTrackVSel05SimMuonAssoc.ptMinTP = 5.0
looseModMuonTrackVSel05SimMuonAssoc.usetracker = True
looseModMuonTrackVSel05SimMuonAssoc.usemuon = True
looseModMuonTrackVSel05SimMuonAssoc.useMCTruth = False
looseModMuonTrackVSel05SimMuonAssoc.dirName = 'Muons/RecoMuonV/MultiTrack/Cut5/'

looseModMuonTrackVSel05MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
looseModMuonTrackVSel05MuonAssoc.associatormap = 'tpToLooseModSel05MuonAssociation'
looseModMuonTrackVSel05MuonAssoc.associators = ('MuonAssociationByHits',)
looseModMuonTrackVSel05MuonAssoc.label = ('bestMuonLooseMod5',)
looseModMuonTrackVSel05MuonAssoc.ptMinTP = 5.0
looseModMuonTrackVSel05MuonAssoc.usetracker = True
looseModMuonTrackVSel05MuonAssoc.usemuon = True
looseModMuonTrackVSel05MuonAssoc.useMCTruth = False

#-------------------------------------------------------------------------------------------------------------------

looseMuonTrackVSel3MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
looseMuonTrackVSel3MuonAssoc.associatormap = 'tpToLooseSel3MuonAssociation'
looseMuonTrackVSel3MuonAssoc.associators = ('MuonAssociationByHits',)
looseMuonTrackVSel3MuonAssoc.label = ('bestMuonLoose',)
looseMuonTrackVSel3MuonAssoc.usetracker = True
looseMuonTrackVSel3MuonAssoc.usemuon = True
looseMuonTrackVSel3MuonAssoc.useMCTruth = False

looseMuonTrackVSel35SimMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
looseMuonTrackVSel35SimMuonAssoc.associatormap = 'tpToLooseSel3MuonAssociation'
looseMuonTrackVSel35SimMuonAssoc.associators = ('MuonAssociationByHits',)
looseMuonTrackVSel35SimMuonAssoc.label = ('bestMuonLoose',)
looseMuonTrackVSel35SimMuonAssoc.ptMinTP = 5.0
looseMuonTrackVSel35SimMuonAssoc.usetracker = True
looseMuonTrackVSel35SimMuonAssoc.usemuon = True
looseMuonTrackVSel35SimMuonAssoc.useMCTruth = False
looseMuonTrackVSel35SimMuonAssoc.dirName = 'Muons/RecoMuonV/MultiTrack/Cut5/'

looseMuonTrackVSel35MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
looseMuonTrackVSel35MuonAssoc.associatormap = 'tpToLooseSel35MuonAssociation'
looseMuonTrackVSel35MuonAssoc.associators = ('MuonAssociationByHits',)
looseMuonTrackVSel35MuonAssoc.label = ('bestMuonLoose5',)
looseMuonTrackVSel35MuonAssoc.ptMinTP = 5.0
looseMuonTrackVSel35MuonAssoc.usetracker = True
looseMuonTrackVSel35MuonAssoc.usemuon = True
looseMuonTrackVSel35MuonAssoc.useMCTruth = False

looseMuonTrackVSel4MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
looseMuonTrackVSel4MuonAssoc.associatormap = 'tpToLooseSel4MuonAssociation'
looseMuonTrackVSel4MuonAssoc.associators = ('MuonAssociationByHits',)
looseMuonTrackVSel4MuonAssoc.label = ('bestMuonLoose',)
looseMuonTrackVSel4MuonAssoc.usetracker = True
looseMuonTrackVSel4MuonAssoc.usemuon = True

#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------

tightMuonTrackVSelMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tightMuonTrackVSelMuonAssoc.associatormap = 'tpToTightSelMuonAssociation'
tightMuonTrackVSelMuonAssoc.associators = ('MuonAssociationByHits',)
tightMuonTrackVSelMuonAssoc.label = ('bestMuonTight',)
tightMuonTrackVSelMuonAssoc.usetracker = True
tightMuonTrackVSelMuonAssoc.usemuon = True

tightMuonTrackVSel2MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tightMuonTrackVSel2MuonAssoc.associatormap = 'tpToTightSel2MuonAssociation'
tightMuonTrackVSel2MuonAssoc.associators = ('MuonAssociationByHits',)
tightMuonTrackVSel2MuonAssoc.label = ('bestMuonTight',)
tightMuonTrackVSel2MuonAssoc.usetracker = True
tightMuonTrackVSel2MuonAssoc.usemuon = True

tightMuonTrackVSel0MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tightMuonTrackVSel0MuonAssoc.associatormap = 'tpToTightSel0MuonAssociation'
tightMuonTrackVSel0MuonAssoc.associators = ('MuonAssociationByHits',)
tightMuonTrackVSel0MuonAssoc.label = ('bestMuonTight',)
tightMuonTrackVSel0MuonAssoc.usetracker = True
tightMuonTrackVSel0MuonAssoc.usemuon = True
tightMuonTrackVSel0MuonAssoc.useMCTruth = False

tightMuonTrackVSel0BSMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tightMuonTrackVSel0BSMuonAssoc.associatormap = 'tpToTightSel0BSMuonAssociation'
tightMuonTrackVSel0BSMuonAssoc.associators = ('MuonAssociationByHits',)
tightMuonTrackVSel0BSMuonAssoc.label = ('bestMuonTightBS',)
tightMuonTrackVSel0BSMuonAssoc.usetracker = True
tightMuonTrackVSel0BSMuonAssoc.usemuon = True
tightMuonTrackVSel0BSMuonAssoc.useMCTruth = False

tightMuonTrackVSelUncMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tightMuonTrackVSelUncMuonAssoc.associatormap = 'tpToTightSelUncMuonAssociation'
tightMuonTrackVSelUncMuonAssoc.associators = ('MuonAssociationByHits',)
tightMuonTrackVSelUncMuonAssoc.label = ('bestMuonTight',)
tightMuonTrackVSelUncMuonAssoc.usetracker = False
tightMuonTrackVSelUncMuonAssoc.usemuon = True
tightMuonTrackVSelUncMuonAssoc.useMCTruth = False

tightMuonTrackVSel05SimMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tightMuonTrackVSel05SimMuonAssoc.associatormap = 'tpToTightSel0MuonAssociation'
tightMuonTrackVSel05SimMuonAssoc.associators = ('MuonAssociationByHits',)
tightMuonTrackVSel05SimMuonAssoc.label = ('bestMuonTight',)
tightMuonTrackVSel05SimMuonAssoc.ptMinTP = 5.0
tightMuonTrackVSel05SimMuonAssoc.usetracker = True
tightMuonTrackVSel05SimMuonAssoc.usemuon = True
tightMuonTrackVSel05SimMuonAssoc.useMCTruth = False
tightMuonTrackVSel05SimMuonAssoc.dirName = 'Muons/RecoMuonV/MultiTrack/Cut5/'

tightMuonTrackVSel05SimBSMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tightMuonTrackVSel05SimBSMuonAssoc.associatormap = 'tpToTightSel0BSMuonAssociation'
tightMuonTrackVSel05SimBSMuonAssoc.associators = ('MuonAssociationByHits',)
tightMuonTrackVSel05SimBSMuonAssoc.label = ('bestMuonTightBS',)
tightMuonTrackVSel05SimBSMuonAssoc.ptMinTP = 5.0
tightMuonTrackVSel05SimBSMuonAssoc.usetracker = True
tightMuonTrackVSel05SimBSMuonAssoc.usemuon = True
tightMuonTrackVSel05SimBSMuonAssoc.useMCTruth = False
tightMuonTrackVSel05SimBSMuonAssoc.dirName = 'Muons/RecoMuonV/MultiTrack/Cut5/'

tightMuonTrackVSel05MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tightMuonTrackVSel05MuonAssoc.associatormap = 'tpToTightSel05MuonAssociation'
tightMuonTrackVSel05MuonAssoc.associators = ('MuonAssociationByHits',)
tightMuonTrackVSel05MuonAssoc.label = ('bestMuonTight5',)
tightMuonTrackVSel05MuonAssoc.ptMinTP = 5.0
tightMuonTrackVSel05MuonAssoc.usetracker = True
tightMuonTrackVSel05MuonAssoc.usemuon = True
tightMuonTrackVSel05MuonAssoc.useMCTruth = False

tightMuonTrackVSel05BSMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tightMuonTrackVSel05BSMuonAssoc.associatormap = 'tpToTightSel05BSMuonAssociation'
tightMuonTrackVSel05BSMuonAssoc.associators = ('MuonAssociationByHits',)
tightMuonTrackVSel05BSMuonAssoc.label = ('bestMuonTightBS5',)
tightMuonTrackVSel05BSMuonAssoc.ptMinTP = 5.0
tightMuonTrackVSel05BSMuonAssoc.usetracker = True
tightMuonTrackVSel05BSMuonAssoc.usemuon = True
tightMuonTrackVSel05BSMuonAssoc.useMCTruth = False

tightMuonTrackVSel3MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tightMuonTrackVSel3MuonAssoc.associatormap = 'tpToTightSel3MuonAssociation'
tightMuonTrackVSel3MuonAssoc.associators = ('MuonAssociationByHits',)
tightMuonTrackVSel3MuonAssoc.label = ('bestMuonTight',)
tightMuonTrackVSel3MuonAssoc.usetracker = True
tightMuonTrackVSel3MuonAssoc.usemuon = True
tightMuonTrackVSel3MuonAssoc.useMCTruth = False

tightMuonTrackVSel35SimMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tightMuonTrackVSel35SimMuonAssoc.associatormap = 'tpToTightSel3MuonAssociation'
tightMuonTrackVSel35SimMuonAssoc.associators = ('MuonAssociationByHits',)
tightMuonTrackVSel35SimMuonAssoc.label = ('bestMuonTight',)
tightMuonTrackVSel35SimMuonAssoc.ptMinTP = 5.0
tightMuonTrackVSel35SimMuonAssoc.usetracker = True
tightMuonTrackVSel35SimMuonAssoc.usemuon = True
tightMuonTrackVSel35SimMuonAssoc.useMCTruth = False
tightMuonTrackVSel35SimMuonAssoc.dirName = 'Muons/RecoMuonV/MultiTrack/Cut5/'

tightMuonTrackVSel35MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tightMuonTrackVSel35MuonAssoc.associatormap = 'tpToTightSel35MuonAssociation'
tightMuonTrackVSel35MuonAssoc.associators = ('MuonAssociationByHits',)
tightMuonTrackVSel35MuonAssoc.label = ('bestMuonTight5',)
tightMuonTrackVSel35MuonAssoc.ptMinTP = 5.0
tightMuonTrackVSel35MuonAssoc.usetracker = True
tightMuonTrackVSel35MuonAssoc.usemuon = True
tightMuonTrackVSel35MuonAssoc.useMCTruth = False

tightMuonTrackVSel4MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tightMuonTrackVSel4MuonAssoc.associatormap = 'tpToTightSel4MuonAssociation'
tightMuonTrackVSel4MuonAssoc.associators = ('MuonAssociationByHits',)
tightMuonTrackVSel4MuonAssoc.label = ('bestMuonTight',)
tightMuonTrackVSel4MuonAssoc.usetracker = True
tightMuonTrackVSel4MuonAssoc.usemuon = True

#-------------------------------------------------------------------------------------------------------------------

tightModMuonTrackVSel0MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tightModMuonTrackVSel0MuonAssoc.associatormap = 'tpToTightModSel0MuonAssociation'
tightModMuonTrackVSel0MuonAssoc.associators = ('MuonAssociationByHits',)
tightModMuonTrackVSel0MuonAssoc.label = ('bestMuonTightMod',)
tightModMuonTrackVSel0MuonAssoc.usetracker = True
tightModMuonTrackVSel0MuonAssoc.usemuon = True
tightModMuonTrackVSel0MuonAssoc.useMCTruth = False

tightModMuonTrackVSelUncMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tightModMuonTrackVSelUncMuonAssoc.associatormap = 'tpToTightModSelUncMuonAssociation'
tightModMuonTrackVSelUncMuonAssoc.associators = ('MuonAssociationByHits',)
tightModMuonTrackVSelUncMuonAssoc.label = ('bestMuonTightMod',)
tightModMuonTrackVSelUncMuonAssoc.usetracker = False
tightModMuonTrackVSelUncMuonAssoc.usemuon = True
tightModMuonTrackVSelUncMuonAssoc.useMCTruth = False

tightModMuonTrackVSel05SimMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tightModMuonTrackVSel05SimMuonAssoc.associatormap = 'tpToTightModSel0MuonAssociation'
tightModMuonTrackVSel05SimMuonAssoc.associators = ('MuonAssociationByHits',)
tightModMuonTrackVSel05SimMuonAssoc.label = ('bestMuonTightMod',)
tightModMuonTrackVSel05SimMuonAssoc.ptMinTP = 5.0
tightModMuonTrackVSel05SimMuonAssoc.usetracker = True
tightModMuonTrackVSel05SimMuonAssoc.usemuon = True
tightModMuonTrackVSel05SimMuonAssoc.useMCTruth = False
tightModMuonTrackVSel05SimMuonAssoc.dirName = 'Muons/RecoMuonV/MultiTrack/Cut5/'

tightModMuonTrackVSel05MuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tightModMuonTrackVSel05MuonAssoc.associatormap = 'tpToTightModSel05MuonAssociation'
tightModMuonTrackVSel05MuonAssoc.associators = ('MuonAssociationByHits',)
tightModMuonTrackVSel05MuonAssoc.label = ('bestMuonTightMod5',)
tightModMuonTrackVSel05MuonAssoc.ptMinTP = 5.0
tightModMuonTrackVSel05MuonAssoc.usetracker = True
tightModMuonTrackVSel05MuonAssoc.usemuon = True
tightModMuonTrackVSel05MuonAssoc.useMCTruth = False

displacedStaMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
displacedStaMuonTrackVMuonAssoc.associatormap = 'tpToDisplacedStaMuonAssociation'
displacedStaMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
displacedStaMuonTrackVMuonAssoc.label = ('displacedStandAloneMuons',)
displacedStaMuonTrackVMuonAssoc.usetracker = False
displacedStaMuonTrackVMuonAssoc.usemuon = True
displacedStaMuonTrackVMuonAssoc.tipTP = 300000.
displacedStaMuonTrackVMuonAssoc.lipTP = 300000.
displacedStaMuonTrackVMuonAssoc.vertexSrc = ""
displacedStaMuonTrackVMuonAssoc.stableOnlyTP = False

displacedStaSeedTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
displacedStaSeedTrackVMuonAssoc.associatormap = 'tpToDisplacedStaSeedAssociation'
displacedStaSeedTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
displacedStaSeedTrackVMuonAssoc.label = ('seedsOfDisplacedSTAmuons',)
displacedStaSeedTrackVMuonAssoc.usetracker = False
displacedStaSeedTrackVMuonAssoc.usemuon = True

##################################################################################

# Configurations for RecoMuonValidators
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from Validation.RecoMuon.RecoMuonValidator_cfi import *

#import SimGeneral.MixingModule.mixNoPU_cfi
from SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi import *
from SimMuon.MCTruth.MuonAssociatorByHits_cfi import muonAssociatorByHitsCommonParameters

#tracker
muonAssociatorByHitsESProducerNoSimHits_trk = SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi.muonAssociatorByHitsESProducerNoSimHits.clone()
muonAssociatorByHitsESProducerNoSimHits_trk.ComponentName = 'muonAssociatorByHits_NoSimHits_tracker'
muonAssociatorByHitsESProducerNoSimHits_trk.UseTracker = True
muonAssociatorByHitsESProducerNoSimHits_trk.UseMuon  = False
recoMuonVMuAssoc_trk = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_trk.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Trk'
recoMuonVMuAssoc_trk.simLabel = 'mix:MergedTrackTruth'
recoMuonVMuAssoc_trk.muAssocLabel = 'muonAssociatorByHits_NoSimHits_tracker'
recoMuonVMuAssoc_trk.trackType = 'inner'
recoMuonVMuAssoc_trk.selection = "isTrackerMuon"

#tracker and PF
muonAssociatorByHitsESProducerNoSimHits_trkPF = SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi.muonAssociatorByHitsESProducerNoSimHits.clone()
muonAssociatorByHitsESProducerNoSimHits_trkPF.ComponentName = 'muonAssociatorByHits_NoSimHits_trackerPF'
muonAssociatorByHitsESProducerNoSimHits_trkPF.UseTracker = True
muonAssociatorByHitsESProducerNoSimHits_trkPF.UseMuon  = False
recoMuonVMuAssoc_trkPF = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_trkPF.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_TrkPF'
recoMuonVMuAssoc_trkPF.usePFMuon = True
recoMuonVMuAssoc_trkPF.simLabel = 'mix:MergedTrackTruth'
recoMuonVMuAssoc_trkPF.muAssocLabel = 'muonAssociatorByHits_NoSimHits_trackerPF'
recoMuonVMuAssoc_trkPF.trackType = 'inner'
recoMuonVMuAssoc_trkPF.selection = "isTrackerMuon & isPFMuon"

#standalone
muonAssociatorByHitsESProducerNoSimHits_sta = SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi.muonAssociatorByHitsESProducerNoSimHits.clone()
muonAssociatorByHitsESProducerNoSimHits_sta.ComponentName = 'muonAssociatorByHits_NoSimHits_standalone'
muonAssociatorByHitsESProducerNoSimHits_sta.UseTracker = False
muonAssociatorByHitsESProducerNoSimHits_sta.UseMuon  = True
recoMuonVMuAssoc_sta = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_sta.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Sta'
recoMuonVMuAssoc_sta.simLabel = 'mix:MergedTrackTruth'
recoMuonVMuAssoc_sta.muAssocLabel = 'muonAssociatorByHits_NoSimHits_standalone'
recoMuonVMuAssoc_sta.trackType = 'outer'
recoMuonVMuAssoc_sta.selection = "isStandAloneMuon"

#seed of StandAlone
muonAssociatorByHitsESProducerNoSimHits_Seedsta = SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi.muonAssociatorByHitsESProducerNoSimHits.clone()
muonAssociatorByHitsESProducerNoSimHits_Seedsta.ComponentName = 'muonAssociatorByHits_NoSimHits_seedOfStandalone'
muonAssociatorByHitsESProducerNoSimHits_Seedsta.UseTracker = False
muonAssociatorByHitsESProducerNoSimHits_Seedsta.UseMuon  = True
recoMuonVMuAssoc_seedSta = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_seedSta.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_SeedSta'
recoMuonVMuAssoc_seedSta.simLabel = 'mix:MergedTrackTruth'
recoMuonVMuAssoc_seedSta.muAssocLabel = 'muonAssociatorByHits_NoSimHits_standalone'
recoMuonVMuAssoc_seedSta.trackType = 'outer'
recoMuonVMuAssoc_seedSta.selection = ""

#standalone and PF
muonAssociatorByHitsESProducerNoSimHits_staPF = SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi.muonAssociatorByHitsESProducerNoSimHits.clone()
muonAssociatorByHitsESProducerNoSimHits_staPF.ComponentName = 'muonAssociatorByHits_NoSimHits_standalonePF'
muonAssociatorByHitsESProducerNoSimHits_staPF.UseTracker = False
muonAssociatorByHitsESProducerNoSimHits_staPF.UseMuon  = True
recoMuonVMuAssoc_staPF = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_staPF.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_StaPF'
recoMuonVMuAssoc_staPF.usePFMuon = True
recoMuonVMuAssoc_staPF.simLabel = 'mix:MergedTrackTruth'
recoMuonVMuAssoc_staPF.muAssocLabel = 'muonAssociatorByHits_NoSimHits_standalonePF'
recoMuonVMuAssoc_staPF.trackType = 'outer'
recoMuonVMuAssoc_staPF.selection = "isStandAloneMuon & isPFMuon"

#global
muonAssociatorByHitsESProducerNoSimHits_glb = SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi.muonAssociatorByHitsESProducerNoSimHits.clone()
muonAssociatorByHitsESProducerNoSimHits_glb.ComponentName = 'muonAssociatorByHits_NoSimHits_global'
muonAssociatorByHitsESProducerNoSimHits_glb.UseTracker = True
muonAssociatorByHitsESProducerNoSimHits_glb.UseMuon  = True
recoMuonVMuAssoc_glb = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_glb.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Glb'
recoMuonVMuAssoc_glb.simLabel = 'mix:MergedTrackTruth'
recoMuonVMuAssoc_glb.muAssocLabel = 'muonAssociatorByHits_NoSimHits_global'
recoMuonVMuAssoc_glb.trackType = 'global'
recoMuonVMuAssoc_glb.selection = "isGlobalMuon"

#global and PF
muonAssociatorByHitsESProducerNoSimHits_glbPF = SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi.muonAssociatorByHitsESProducerNoSimHits.clone()
muonAssociatorByHitsESProducerNoSimHits_glbPF.ComponentName = 'muonAssociatorByHits_NoSimHits_globalPF'
muonAssociatorByHitsESProducerNoSimHits_glbPF.UseTracker = True
muonAssociatorByHitsESProducerNoSimHits_glbPF.UseMuon  = True
recoMuonVMuAssoc_glbPF = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_glbPF.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_GlbPF'
recoMuonVMuAssoc_glbPF.usePFMuon = True
recoMuonVMuAssoc_glbPF.simLabel = 'mix:MergedTrackTruth'
recoMuonVMuAssoc_glbPF.muAssocLabel = 'muonAssociatorByHits_NoSimHits_globalPF'
recoMuonVMuAssoc_glbPF.trackType = 'global'
recoMuonVMuAssoc_glbPF.selection = "isGlobalMuon & isPFMuon"

#tight
muonAssociatorByHitsESProducerNoSimHits_tgt = SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi.muonAssociatorByHitsESProducerNoSimHits.clone()
muonAssociatorByHitsESProducerNoSimHits_tgt.ComponentName = 'muonAssociatorByHits_NoSimHits_tight'
muonAssociatorByHitsESProducerNoSimHits_tgt.UseTracker = True
muonAssociatorByHitsESProducerNoSimHits_tgt.UseMuon  = True
recoMuonVMuAssoc_tgt = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_tgt.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Tgt'
recoMuonVMuAssoc_tgt.simLabel = 'mix:MergedTrackTruth'
recoMuonVMuAssoc_tgt.muAssocLabel = 'muonAssociatorByHits_NoSimHits_tight'
recoMuonVMuAssoc_tgt.trackType = 'global'
recoMuonVMuAssoc_tgt.selection = 'isGlobalMuon'
recoMuonVMuAssoc_tgt.wantTightMuon = True
recoMuonVMuAssoc_tgt.beamSpot = 'offlineBeamSpot'
recoMuonVMuAssoc_tgt.primaryVertex = 'offlinePrimaryVertices'

# Muon validation sequence

muonValidation_seq = cms.Sequence(
        #trkProbeTrackVMuonAssoc+trkMuonTrackVTrackAssoc
#       + staSeedTrackVMuonAssoc
#	+ staMuonTrackVMuonAssoc + 
	#staUpdMuonTrackVMuonAssoc
	#+staUpdMuonTrackVMuonAssoc2St+
	#staUpdMuonTrackVMuonAssoc2StTime
	#+ glbMuonTrackVMuonAssoc
# 
	#+ trkProbeTrackVSelMuonAssoc
	#+ staUpdMuonTrackVSelMuonAssoc+staUpdMuonTrackVSel2MuonAssoc
	#+ staUpdMuonTrackVSelMuonAssoc2St
	#+ staUpdMuonTrackVSel2MuonAssoc2St
	#+ staUpdMuonTrackVSelMuonAssoc2StTime
 	staUpdMuonTrackVSel2MuonAssoc2StTime
    +staUpdMuonTrackVSel2MuonAssoc2StTime05
	#+ staUpd10SelMuonTrackVMuonAssoc+staUpd20SelMuonTrackVMuonAssoc
	#+ glbMuonTrackVSelMuonAssoc+glbMuonTrackVSel2MuonAssoc+glbMuonTrackVSel3MuonAssoc+glbMuonTrackVSel4MuonAssoc
	#+ looseMuonTrackVSelMuonAssoc+looseMuonTrackVSel2MuonAssoc
	#+ looseMuonTrackVSel3MuonAssoc 
	+ looseMuonTrackVSel0MuonAssoc
	+ looseMuonTrackVSelUncMuonAssoc
	#+looseMuonTrackVSel4MuonAssoc
	#+ looseMuonTrackVSel35MuonAssoc
 	+ looseMuonTrackVSel05MuonAssoc
	#+ looseMuonTrackVSel35SimMuonAssoc 
	+ looseMuonTrackVSel05SimMuonAssoc
	#+ tightMuonTrackVSelMuonAssoc+tightMuonTrackVSel2MuonAssoc
	#+ tightMuonTrackVSel3MuonAssoc 
	+ tightMuonTrackVSel0MuonAssoc
	+ tightMuonTrackVSelUncMuonAssoc
	#+tightMuonTrackVSel4MuonAssoc
	#+ tightMuonTrackVSel35MuonAssoc 
	+ tightMuonTrackVSel05MuonAssoc
	#+ tightMuonTrackVSel35SimMuonAssoc 
	+ tightMuonTrackVSel05SimMuonAssoc
    #+ tightMuonTrackVSel0BSMuonAssoc
    #+ tightMuonTrackVSel05BSMuonAssoc
    #+ tightMuonTrackVSel05SimBSMuonAssoc
	+ looseModMuonTrackVSel0MuonAssoc
	+ looseModMuonTrackVSelUncMuonAssoc
 	+ looseModMuonTrackVSel05MuonAssoc
	+ looseModMuonTrackVSel05SimMuonAssoc
	+ tightModMuonTrackVSel0MuonAssoc
	+ tightModMuonTrackVSelUncMuonAssoc
 	+ tightModMuonTrackVSel05MuonAssoc
	+ tightModMuonTrackVSel05SimMuonAssoc
#
#	+ recoMuonVMuAssoc_trk+recoMuonVMuAssoc_sta+recoMuonVMuAssoc_glb+recoMuonVMuAssoc_tgt
	#+ bestMuonLooseTrackVTrackAssoc #+ bestMuonLooseTrackVTrackAssoc5 + bestMuonLooseTrackVTrackAssoc10 + bestMuonLooseTrackVTrackAssoc20
#	+ bestMuonLoose2TrackVTrackAssoc + bestMuonLoose2TrackVTrackAssoc5 + bestMuonLoose2TrackVTrackAssoc10 + bestMuonLoose2TrackVTrackAssoc20
	#+ bestMuonTightTrackVTrackAssoc #+ bestMuonTightTrackVTrackAssoc5 + bestMuonTightTrackVTrackAssoc10 + bestMuonTightTrackVTrackAssoc20

)
                                  
muonValidationTEV_seq = cms.Sequence(tevMuonFirstTrackVMuonAssoc+tevMuonPickyTrackVMuonAssoc+tevMuonDytTrackVMuonAssoc)

muonValidationRefit_seq = cms.Sequence(staRefitMuonTrackVMuonAssoc+staRefitUpdMuonTrackVMuonAssoc)

muonValidationSET_seq = cms.Sequence(staSETMuonTrackVMuonAssoc+staSETUpdMuonTrackVMuonAssoc+glbSETMuonTrackVMuonAssoc)

muonValidationCosmic_seq = cms.Sequence(trkCosmicMuonTrackVTrackAssoc
                                 +staCosmicMuonTrackVMuonAssoc+glbCosmicMuonTrackVMuonAssoc)

#muonValidationDisplaced_seq = cms.Sequence(staMuonTrackVMuonAssoc+staUpdMuonTrackVMuonAssoc+displacedStaMuonTrackVMuonAssoc)
muonValidationDisplaced_seq = cms.Sequence(displacedStaMuonTrackVMuonAssoc)


# The muon association and validation sequence

recoMuonValidation = cms.Sequence(
                                  #(muonAssociation_seq*muonValidation_seq)
                                  #+(muonAssociationTEV_seq*muonValidationTEV_seq)
                                  #+(muonAssociationSET_seq*muonValidationSET_seq)
                                  #(muonAssociationRefit_seq*muonValidationRefit_seq)+
                                  (muonAssociationDisplaced_seq*muonValidationDisplaced_seq)
                                 )

recoCosmicMuonValidation = cms.Sequence(muonAssociationCosmic_seq*muonValidationCosmic_seq)
