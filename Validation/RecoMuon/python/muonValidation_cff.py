import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *
# Configurations for MuonTrackValidators
import Validation.RecoMuon.MuonTrackValidator_cfi

TrackAssociatorByChi2ESProducer.chi2cut = cms.double(500.0)
TrackAssociatorByPullESProducer = TrackAssociatorByChi2ESProducer.clone(chi2cut = 50.0,onlyDiagonal = True,ComponentName = 'TrackAssociatorByPull')

trkMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trkMuonTrackVTrackAssoc.associatormap = 'tpToTkmuTrackAssociation'
trkMuonTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
#trkMuonTrackVTrackAssoc.label = ('generalTracks',)
trkMuonTrackVTrackAssoc.label = ('probeTracks',)
trkMuonTrackVTrackAssoc.usetracker = True
trkMuonTrackVTrackAssoc.usemuon = False

bestMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonTrackVTrackAssoc.associatormap = ''
bestMuonTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonTrackVTrackAssoc.label = ('bestMuon',)
bestMuonTrackVTrackAssoc.usetracker = True
bestMuonTrackVTrackAssoc.usemuon = True

bestMuonTrackVTrackAssoc3 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonTrackVTrackAssoc3.associatormap = ''
bestMuonTrackVTrackAssoc3.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonTrackVTrackAssoc3.label = ('bestMuon3',)
bestMuonTrackVTrackAssoc3.usetracker = True
bestMuonTrackVTrackAssoc3.usemuon = True

bestMuonTrackVTrackAssoc5 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonTrackVTrackAssoc5.associatormap = ''
bestMuonTrackVTrackAssoc5.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonTrackVTrackAssoc5.label = ('bestMuon5',)
bestMuonTrackVTrackAssoc5.usetracker = True
bestMuonTrackVTrackAssoc5.usemuon = True

bestMuonTrackVTrackAssoc10 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonTrackVTrackAssoc10.associatormap = ''
bestMuonTrackVTrackAssoc10.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonTrackVTrackAssoc10.label = ('bestMuon10',)
bestMuonTrackVTrackAssoc10.usetracker = True
bestMuonTrackVTrackAssoc10.usemuon = True

bestMuonTrackVTrackAssoc15 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonTrackVTrackAssoc15.associatormap = ''
bestMuonTrackVTrackAssoc15.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonTrackVTrackAssoc15.label = ('bestMuon15',)
bestMuonTrackVTrackAssoc15.usetracker = True
bestMuonTrackVTrackAssoc15.usemuon = True

bestMuonTrackVTrackAssoc20 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonTrackVTrackAssoc20.associatormap = ''
bestMuonTrackVTrackAssoc20.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonTrackVTrackAssoc20.label = ('bestMuon20',)
bestMuonTrackVTrackAssoc20.usetracker = True
bestMuonTrackVTrackAssoc20.usemuon = True

bestMuonTrackVTrackAssoc50 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonTrackVTrackAssoc50.associatormap = ''
bestMuonTrackVTrackAssoc50.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonTrackVTrackAssoc50.label = ('bestMuon50',)
bestMuonTrackVTrackAssoc50.usetracker = True
bestMuonTrackVTrackAssoc50.usemuon = True

bestMuonLooseTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonLooseTrackVTrackAssoc.associatormap = ''
bestMuonLooseTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonLooseTrackVTrackAssoc.label = ('bestMuonLoose',)
bestMuonLooseTrackVTrackAssoc.usetracker = True
bestMuonLooseTrackVTrackAssoc.usemuon = True

bestMuonLooseTrackVTrackAssoc3 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonLooseTrackVTrackAssoc3.associatormap = ''
bestMuonLooseTrackVTrackAssoc3.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonLooseTrackVTrackAssoc3.label = ('bestMuonLoose3',)
bestMuonLooseTrackVTrackAssoc3.usetracker = True
bestMuonLooseTrackVTrackAssoc3.usemuon = True

bestMuonLooseTrackVTrackAssoc5 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonLooseTrackVTrackAssoc5.associatormap = ''
bestMuonLooseTrackVTrackAssoc5.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonLooseTrackVTrackAssoc5.label = ('bestMuonLoose5',)
bestMuonLooseTrackVTrackAssoc5.usetracker = True
bestMuonLooseTrackVTrackAssoc5.usemuon = True

bestMuonLooseTrackVTrackAssoc10 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonLooseTrackVTrackAssoc10.associatormap = ''
bestMuonLooseTrackVTrackAssoc10.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonLooseTrackVTrackAssoc10.label = ('bestMuonLoose10',)
bestMuonLooseTrackVTrackAssoc10.usetracker = True
bestMuonLooseTrackVTrackAssoc10.usemuon = True

bestMuonLooseTrackVTrackAssoc15 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonLooseTrackVTrackAssoc15.associatormap = ''
bestMuonLooseTrackVTrackAssoc15.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonLooseTrackVTrackAssoc15.label = ('bestMuonLoose15',)
bestMuonLooseTrackVTrackAssoc15.usetracker = True
bestMuonLooseTrackVTrackAssoc15.usemuon = True

bestMuonLooseTrackVTrackAssoc20 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonLooseTrackVTrackAssoc20.associatormap = ''
bestMuonLooseTrackVTrackAssoc20.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonLooseTrackVTrackAssoc20.label = ('bestMuonLoose20',)
bestMuonLooseTrackVTrackAssoc20.usetracker = True
bestMuonLooseTrackVTrackAssoc20.usemuon = True

bestMuonTightTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonTightTrackVTrackAssoc.associatormap = ''
bestMuonTightTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonTightTrackVTrackAssoc.label = ('bestMuonTight',)
bestMuonTightTrackVTrackAssoc.usetracker = True
bestMuonTightTrackVTrackAssoc.usemuon = True

bestMuonTightTrackVTrackAssoc3 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonTightTrackVTrackAssoc3.associatormap = ''
bestMuonTightTrackVTrackAssoc3.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonTightTrackVTrackAssoc3.label = ('bestMuonTight3',)
bestMuonTightTrackVTrackAssoc3.usetracker = True
bestMuonTightTrackVTrackAssoc3.usemuon = True

bestMuonTightTrackVTrackAssoc5 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonTightTrackVTrackAssoc5.associatormap = ''
bestMuonTightTrackVTrackAssoc5.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonTightTrackVTrackAssoc5.label = ('bestMuonTight5',)
bestMuonTightTrackVTrackAssoc5.usetracker = True
bestMuonTightTrackVTrackAssoc5.usemuon = True

bestMuonTightTrackVTrackAssoc10 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonTightTrackVTrackAssoc10.associatormap = ''
bestMuonTightTrackVTrackAssoc10.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonTightTrackVTrackAssoc10.label = ('bestMuonTight10',)
bestMuonTightTrackVTrackAssoc10.usetracker = True
bestMuonTightTrackVTrackAssoc10.usemuon = True

bestMuonTightTrackVTrackAssoc15 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonTightTrackVTrackAssoc15.associatormap = ''
bestMuonTightTrackVTrackAssoc15.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonTightTrackVTrackAssoc15.label = ('bestMuonTight15',)
bestMuonTightTrackVTrackAssoc15.usetracker = True
bestMuonTightTrackVTrackAssoc15.usemuon = True

bestMuonTightTrackVTrackAssoc20 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonTightTrackVTrackAssoc20.associatormap = ''
bestMuonTightTrackVTrackAssoc20.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonTightTrackVTrackAssoc20.label = ('bestMuonTight20',)
bestMuonTightTrackVTrackAssoc20.usetracker = True
bestMuonTightTrackVTrackAssoc20.usemuon = True

bestMuonTunePTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
bestMuonTunePTrackVTrackAssoc.associatormap = ''
bestMuonTunePTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
bestMuonTunePTrackVTrackAssoc.label = ('bestMuonTuneP',)
bestMuonTunePTrackVTrackAssoc.usetracker = True
bestMuonTunePTrackVTrackAssoc.usemuon = True

trackerMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trackerMuonTrackVTrackAssoc.associatormap = ''
trackerMuonTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
trackerMuonTrackVTrackAssoc.label = ('trackerMuons',)
trackerMuonTrackVTrackAssoc.usetracker = True
trackerMuonTrackVTrackAssoc.usemuon = False

trackerMuonTMOneTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trackerMuonTMOneTrackVTrackAssoc.associatormap = ''
trackerMuonTMOneTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
trackerMuonTMOneTrackVTrackAssoc.label = ('TMOneStationTight',)
trackerMuonTMOneTrackVTrackAssoc.usetracker = True
trackerMuonTMOneTrackVTrackAssoc.usemuon = False

trackerMuonTMLastTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trackerMuonTMLastTrackVTrackAssoc.associatormap = ''
trackerMuonTMLastTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
trackerMuonTMLastTrackVTrackAssoc.label = ('TMLastStationAngTight',)
trackerMuonTMLastTrackVTrackAssoc.usetracker = True
trackerMuonTMLastTrackVTrackAssoc.usemuon = False

trackerMuonArbTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trackerMuonArbTrackVTrackAssoc.associatormap = ''
trackerMuonArbTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
trackerMuonArbTrackVTrackAssoc.label = ('TrackerMuonArbitrated',)
trackerMuonArbTrackVTrackAssoc.usetracker = True
trackerMuonArbTrackVTrackAssoc.usemuon = False

trkCosmicMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trkCosmicMuonTrackVTrackAssoc.associatormap = 'tpToTkCosmicTrackAssociation'
trkCosmicMuonTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
trkCosmicMuonTrackVTrackAssoc.label = ('ctfWithMaterialTracksP5LHCNavigation',)
trkCosmicMuonTrackVTrackAssoc.usetracker = True
trkCosmicMuonTrackVTrackAssoc.usemuon = False

staMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staMuonTrackVTrackAssoc.associatormap = 'tpToStaTrackAssociation'
staMuonTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staMuonTrackVTrackAssoc.label = ('standAloneMuons',)
staMuonTrackVTrackAssoc.usetracker = False
staMuonTrackVTrackAssoc.usemuon = True

glbMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVTrackAssoc.associatormap = 'tpToStaTrackAssociation'
glbMuonTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
glbMuonTrackVTrackAssoc.label = ('globalMuons',)
glbMuonTrackVTrackAssoc.usetracker = True
glbMuonTrackVTrackAssoc.usemuon = True

staUpdMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVTrackAssoc.associatormap = 'tpToStaUpdTrackAssociation'
staUpdMuonTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staUpdMuonTrackVTrackAssoc.label = ('standAloneMuons:UpdatedAtVtx',)
staUpdMuonTrackVTrackAssoc.usetracker = False
staUpdMuonTrackVTrackAssoc.usemuon = True

staSETMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staSETMuonTrackVTrackAssoc.associatormap = 'tpToStaSETTrackAssociation'
staSETMuonTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staSETMuonTrackVTrackAssoc.label = ('standAloneSETMuons',)
staSETMuonTrackVTrackAssoc.usetracker = False
staSETMuonTrackVTrackAssoc.usemuon = True

staSETUpdMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staSETUpdMuonTrackVTrackAssoc.associatormap = 'tpToStaSETUpdTrackAssociation'
staSETUpdMuonTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staSETUpdMuonTrackVTrackAssoc.label = ('standAloneSETMuons:UpdatedAtVtx',)
staSETUpdMuonTrackVTrackAssoc.usetracker = False
staSETUpdMuonTrackVTrackAssoc.usemuon = True

glbSETMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbSETMuonTrackVTrackAssoc.associatormap = 'tpToGlbSETTrackAssociation'
glbSETMuonTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
glbSETMuonTrackVTrackAssoc.label = ('globalSETMuons',)
glbSETMuonTrackVTrackAssoc.usetracker = True
glbSETMuonTrackVTrackAssoc.usemuon = True

tevMuonFirstTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonFirstTrackVTrackAssoc.associatormap = 'tpToTevFirstTrackAssociation'
tevMuonFirstTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
tevMuonFirstTrackVTrackAssoc.label = ('tevMuons:firstHit',)
tevMuonFirstTrackVTrackAssoc.usetracker = True
tevMuonFirstTrackVTrackAssoc.usemuon = True

tevMuonPickyTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonPickyTrackVTrackAssoc.associatormap = 'tpToTevPickyTrackAssociation'
tevMuonPickyTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
tevMuonPickyTrackVTrackAssoc.label = ('tevMuons:picky',)
tevMuonPickyTrackVTrackAssoc.usetracker = True
tevMuonPickyTrackVTrackAssoc.usemuon = True

tevMuonDytTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonDytTrackVTrackAssoc.associatormap = 'tpToTevDytTrackAssociation'
tevMuonDytTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
tevMuonDytTrackVTrackAssoc.label = ('tevMuons:dyt',)
tevMuonDytTrackVTrackAssoc.usetracker = True
tevMuonDytTrackVTrackAssoc.usemuon = True

staCosmicMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staCosmicMuonTrackVTrackAssoc.associatormap = 'tpToStaCosmicTrackAssociation'
staCosmicMuonTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staCosmicMuonTrackVTrackAssoc.label = ('cosmicMuons',)
staCosmicMuonTrackVTrackAssoc.usetracker = False
staCosmicMuonTrackVTrackAssoc.usemuon = True

glbCosmicMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbCosmicMuonTrackVTrackAssoc.associatormap = 'tpToGlbCosmicTrackAssociation'
glbCosmicMuonTrackVTrackAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
glbCosmicMuonTrackVTrackAssoc.label = ('globalCosmicMuons',)
glbCosmicMuonTrackVTrackAssoc.usetracker = True
glbCosmicMuonTrackVTrackAssoc.usemuon = True

trkProbeTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
#trkMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trkProbeTrackVMuonAssoc.associatormap = 'tpToTkMuonAssociation' 
trkProbeTrackVMuonAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
##trkMuonTrackVMuonAssoc.label = ('generalTracks',)
trkProbeTrackVMuonAssoc.label = ('probeTracks',)
trkProbeTrackVMuonAssoc.usetracker = True
trkProbeTrackVMuonAssoc.usemuon = False

#staSeedTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
#staSeedTrackVMuonAssoc.associatormap = 'tpToStaSeedAssociation'
#staSeedTrackVMuonAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
#staSeedTrackVMuonAssoc.label = ('seedsOfSTAmuons',)
#staSeedTrackVMuonAssoc.usetracker = False
#staSeedTrackVMuonAssoc.usemuon = True

staMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staMuonTrackVMuonAssoc.associatormap = 'tpToStaMuonAssociation'
staMuonTrackVMuonAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staMuonTrackVMuonAssoc.label = ('standAloneMuons',)
staMuonTrackVMuonAssoc.usetracker = False
staMuonTrackVMuonAssoc.usemuon = True

staUpdMuonTrackVMuonAssoc3 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVMuonAssoc3.associatormap = 'tpToStaMuonAssociation'
staUpdMuonTrackVMuonAssoc3.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staUpdMuonTrackVMuonAssoc3.label = ('staMuonsPt3',)
staUpdMuonTrackVMuonAssoc3.usetracker = False
staUpdMuonTrackVMuonAssoc3.usemuon = True

staUpdMuonTrackVMuonAssoc5 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVMuonAssoc5.associatormap = 'tpToStaMuonAssociation'
staUpdMuonTrackVMuonAssoc5.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staUpdMuonTrackVMuonAssoc5.label = ('staMuonsPt5',)
staUpdMuonTrackVMuonAssoc5.usetracker = False
staUpdMuonTrackVMuonAssoc5.usemuon = True

staUpdMuonTrackVMuonAssoc10 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVMuonAssoc10.associatormap = 'tpToStaMuonAssociation'
staUpdMuonTrackVMuonAssoc10.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staUpdMuonTrackVMuonAssoc10.label = ('staMuonsPt10',)
staUpdMuonTrackVMuonAssoc10.usetracker = False
staUpdMuonTrackVMuonAssoc10.usemuon = True

staUpdMuonTrackVMuonAssoc15 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVMuonAssoc15.associatormap = 'tpToStaMuonAssociation'
staUpdMuonTrackVMuonAssoc15.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staUpdMuonTrackVMuonAssoc15.label = ('staMuonsPt15',)
staUpdMuonTrackVMuonAssoc15.usetracker = False
staUpdMuonTrackVMuonAssoc15.usemuon = True

staUpdMuonTrackVMuonAssoc20 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVMuonAssoc20.associatormap = 'tpToStaMuonAssociation'
staUpdMuonTrackVMuonAssoc20.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staUpdMuonTrackVMuonAssoc20.label = ('staMuonsPt20',)
staUpdMuonTrackVMuonAssoc20.usetracker = False
staUpdMuonTrackVMuonAssoc20.usemuon = True

staUpdMuonTrackVMuonAssoc50 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVMuonAssoc50.associatormap = 'tpToStaMuonAssociation'
staUpdMuonTrackVMuonAssoc50.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staUpdMuonTrackVMuonAssoc50.label = ('staMuonsPt50',)
staUpdMuonTrackVMuonAssoc50.usetracker = False
staUpdMuonTrackVMuonAssoc50.usemuon = True

staUpdMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVMuonAssoc.associatormap = 'tpToStaUpdMuonAssociation'
staUpdMuonTrackVMuonAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staUpdMuonTrackVMuonAssoc.label = ('standAloneMuons:UpdatedAtVtx',)
staUpdMuonTrackVMuonAssoc.usetracker = False
staUpdMuonTrackVMuonAssoc.usemuon = True

glbMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVMuonAssoc.associatormap = 'tpToGlbMuonAssociation'
glbMuonTrackVMuonAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
glbMuonTrackVMuonAssoc.label = ('extractedGlobalMuons',)
glbMuonTrackVMuonAssoc.usetracker = True
glbMuonTrackVMuonAssoc.usemuon = True

glbMuonTrackVMuonAssoc3 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVMuonAssoc3.associatormap = 'tpToGlbMuonAssociation'
glbMuonTrackVMuonAssoc3.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
glbMuonTrackVMuonAssoc3.label = ('extractedGlobalMuons3',)
glbMuonTrackVMuonAssoc3.usetracker = True
glbMuonTrackVMuonAssoc3.usemuon = True

glbMuonTrackVMuonAssoc5 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVMuonAssoc5.associatormap = 'tpToGlbMuonAssociation'
glbMuonTrackVMuonAssoc5.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
glbMuonTrackVMuonAssoc5.label = ('extractedGlobalMuons5',)
glbMuonTrackVMuonAssoc5.usetracker = True
glbMuonTrackVMuonAssoc5.usemuon = True

glbMuonTrackVMuonAssoc10 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVMuonAssoc10.associatormap = 'tpToGlbMuonAssociation'
glbMuonTrackVMuonAssoc10.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
glbMuonTrackVMuonAssoc10.label = ('extractedGlobalMuons10',)
glbMuonTrackVMuonAssoc10.usetracker = True
glbMuonTrackVMuonAssoc10.usemuon = True

glbMuonTrackVMuonAssoc15 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVMuonAssoc15.associatormap = 'tpToGlbMuonAssociation'
glbMuonTrackVMuonAssoc15.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
glbMuonTrackVMuonAssoc15.label = ('extractedGlobalMuons15',)
glbMuonTrackVMuonAssoc15.usetracker = True
glbMuonTrackVMuonAssoc15.usemuon = True

glbMuonTrackVMuonAssoc20 = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVMuonAssoc20.associatormap = 'tpToGlbMuonAssociation'
glbMuonTrackVMuonAssoc20.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
glbMuonTrackVMuonAssoc20.label = ('extractedGlobalMuons20',)
glbMuonTrackVMuonAssoc20.usetracker = True
glbMuonTrackVMuonAssoc20.usemuon = True

staRefitMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staRefitMuonTrackVMuonAssoc.associatormap = 'tpToStaRefitMuonAssociation'
staRefitMuonTrackVMuonAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staRefitMuonTrackVMuonAssoc.label = ('refittedStandAloneMuons',)
staRefitMuonTrackVMuonAssoc.usetracker = False
staRefitMuonTrackVMuonAssoc.usemuon = True

staRefitUpdMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staRefitUpdMuonTrackVMuonAssoc.associatormap = 'tpToStaRefitUpdMuonAssociation'
staRefitUpdMuonTrackVMuonAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staRefitUpdMuonTrackVMuonAssoc.label = ('refittedStandAloneMuons:UpdatedAtVtx',)
staRefitUpdMuonTrackVMuonAssoc.usetracker = False
staRefitUpdMuonTrackVMuonAssoc.usemuon = True

staSETMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staSETMuonTrackVMuonAssoc.associatormap = 'tpToStaSETMuonAssociation'
staSETMuonTrackVMuonAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staSETMuonTrackVMuonAssoc.label = ('standAloneSETMuons',)
staSETMuonTrackVMuonAssoc.usetracker = False
staSETMuonTrackVMuonAssoc.usemuon = True

staSETUpdMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staSETUpdMuonTrackVMuonAssoc.associatormap = 'tpToStaSETUpdMuonAssociation'
staSETUpdMuonTrackVMuonAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staSETUpdMuonTrackVMuonAssoc.label = ('standAloneSETMuons:UpdatedAtVtx',)
staSETUpdMuonTrackVMuonAssoc.usetracker = False
staSETUpdMuonTrackVMuonAssoc.usemuon = True

glbSETMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbSETMuonTrackVMuonAssoc.associatormap = 'tpToGlbSETMuonAssociation'
glbSETMuonTrackVMuonAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
glbSETMuonTrackVMuonAssoc.label = ('globalSETMuons',)
glbSETMuonTrackVMuonAssoc.usetracker = True
glbSETMuonTrackVMuonAssoc.usemuon = True

tevMuonFirstTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonFirstTrackVMuonAssoc.associatormap = 'tpToTevFirstMuonAssociation'
tevMuonFirstTrackVMuonAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
tevMuonFirstTrackVMuonAssoc.label = ('tevMuons:firstHit',)
tevMuonFirstTrackVMuonAssoc.usetracker = True
tevMuonFirstTrackVMuonAssoc.usemuon = True

tevMuonPickyTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonPickyTrackVMuonAssoc.associatormap = 'tpToTevPickyMuonAssociation'
tevMuonPickyTrackVMuonAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
tevMuonPickyTrackVMuonAssoc.label = ('tevMuons:picky',)
tevMuonPickyTrackVMuonAssoc.usetracker = True
tevMuonPickyTrackVMuonAssoc.usemuon = True

tevMuonDytTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonDytTrackVMuonAssoc.associatormap = 'tpToTevDytMuonAssociation'
tevMuonDytTrackVMuonAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
tevMuonDytTrackVMuonAssoc.label = ('tevMuons:dyt',)
tevMuonDytTrackVMuonAssoc.usetracker = True
tevMuonDytTrackVMuonAssoc.usemuon = True

staCosmicMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staCosmicMuonTrackVMuonAssoc.associatormap = 'tpToStaCosmicMuonAssociation'
staCosmicMuonTrackVMuonAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staCosmicMuonTrackVMuonAssoc.label = ('cosmicMuons',)
staCosmicMuonTrackVMuonAssoc.usetracker = False
staCosmicMuonTrackVMuonAssoc.usemuon = True

glbCosmicMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbCosmicMuonTrackVMuonAssoc.associatormap = 'tpToGlbCosmicMuonAssociation'
glbCosmicMuonTrackVMuonAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
glbCosmicMuonTrackVMuonAssoc.label = ('globalCosmicMuons',)
glbCosmicMuonTrackVMuonAssoc.usetracker = True
glbCosmicMuonTrackVMuonAssoc.usemuon = True

### a few more validator modules usable also for Upgrade TP studies
trkProbeTrackVSelMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trkProbeTrackVSelMuonAssoc.associatormap = 'tpToTkSelMuonAssociation'
trkProbeTrackVSelMuonAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
trkProbeTrackVSelMuonAssoc.label = ('probeTracks',)
trkProbeTrackVSelMuonAssoc.usetracker = True
trkProbeTrackVSelMuonAssoc.usemuon = False

staUpdMuonTrackVSelMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVSelMuonAssoc.associatormap = 'tpToStaUpdSelMuonAssociation'
staUpdMuonTrackVSelMuonAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
staUpdMuonTrackVSelMuonAssoc.label = ('standAloneMuons:UpdatedAtVtx',)
staUpdMuonTrackVSelMuonAssoc.usetracker = False
staUpdMuonTrackVSelMuonAssoc.usemuon = True

glbMuonTrackVSelMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVSelMuonAssoc.associatormap = 'tpToGlbSelMuonAssociation'
glbMuonTrackVSelMuonAssoc.associators = ['TrackAssociatorByChi2','TrackAssociatorByPull']
glbMuonTrackVSelMuonAssoc.label = ('extractedGlobalMuons',)
glbMuonTrackVSelMuonAssoc.usetracker = True
glbMuonTrackVSelMuonAssoc.usemuon = True
###


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
	#trkProbeTrackVMuonAssoc+ trkMuonTrackVTrackAssoc
	#+staSeedTrackVMuonAssoc+
	#staMuonTrackVMuonAssoc + 
	staUpdMuonTrackVMuonAssoc + staUpdMuonTrackVMuonAssoc20
	#+ glbMuonTrackVTrackAssoc 
	+ glbMuonTrackVMuonAssoc + glbMuonTrackVMuonAssoc20
	+ bestMuonTrackVTrackAssoc + bestMuonTrackVTrackAssoc20 + bestMuonTrackVTrackAssoc50
	+ bestMuonLooseTrackVTrackAssoc + bestMuonLooseTrackVTrackAssoc20
	+ bestMuonTightTrackVTrackAssoc + bestMuonTightTrackVTrackAssoc20
	#+ bestMuonTunePTrackVTrackAssoc
	+ trackerMuonTrackVTrackAssoc + trackerMuonArbTrackVTrackAssoc 
	+ trackerMuonTMOneTrackVTrackAssoc + trackerMuonTMLastTrackVTrackAssoc 
	#+trkProbeTrackVSelMuonAssoc
	#+ staUpdMuonTrackVSelMuonAssoc + glbMuonTrackVSelMuonAssoc
	)
	#+recoMuonVMuAssoc_trk+recoMuonVMuAssoc_sta+recoMuonVMuAssoc_glb+recoMuonVMuAssoc_tgt)
                                  
muonValidationTEV_seq = cms.Sequence(tevMuonFirstTrackVMuonAssoc+tevMuonPickyTrackVMuonAssoc)#+tevMuonDytTrackVMuonAssoc)

muonValidationRefit_seq = cms.Sequence(staRefitMuonTrackVMuonAssoc+staRefitUpdMuonTrackVMuonAssoc)

muonValidationSET_seq = cms.Sequence(staSETMuonTrackVMuonAssoc+staSETUpdMuonTrackVMuonAssoc+glbSETMuonTrackVMuonAssoc)

muonValidationCosmic_seq = cms.Sequence(trkCosmicMuonTrackVTrackAssoc+staCosmicMuonTrackVMuonAssoc+glbCosmicMuonTrackVMuonAssoc)

# The muon association and validation sequence

recoMuonValidation = cms.Sequence(#probeTracks_seq*
				 (selectedVertices * selectedFirstPrimaryVertex) * 
				 bestMuonTuneP_seq*muonColl_seq*trackColl_seq*extractedMuonTracks_seq*bestMuon_seq*trackerMuon_seq*
				 ((muonValidation_seq)
                                 +(muonValidationTEV_seq)
                                 #+(muonValidationSET_seq)
                                 #+(muonValidationRefit_seq)
				 )
                                 )

recoCosmicMuonValidation = cms.Sequence(muonAssociationCosmic_seq*muonValidationCosmic_seq)
