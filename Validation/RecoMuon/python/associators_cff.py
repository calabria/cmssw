import FWCore.ParameterSet.Config as cms

#Track selector
from Validation.RecoMuon.selectors_cff import *

#TrackAssociation
from SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi import *
import SimTracker.TrackAssociation.quickTrackAssociatorByHits_cfi
import SimTracker.TrackAssociation.TrackAssociatorByPosition_cfi

TrackAssociatorByHits = SimTracker.TrackAssociation.quickTrackAssociatorByHits_cfi.quickTrackAssociatorByHits.clone( ComponentName = 'TrackAssociatorByHits' )

OnlineTrackAssociatorByHits = SimTracker.TrackAssociation.quickTrackAssociatorByHits_cfi.quickTrackAssociatorByHits.clone()
OnlineTrackAssociatorByHits.ComponentName = 'OnlineTrackAssociatorByHits'
OnlineTrackAssociatorByHits.UseGrouped = cms.bool(False)
OnlineTrackAssociatorByHits.UseSplitting = cms.bool(False)
OnlineTrackAssociatorByHits.ThreeHitTracksAreSpecial = False

TrackAssociatorByPosDeltaR = SimTracker.TrackAssociation.TrackAssociatorByPosition_cfi.TrackAssociatorByPosition.clone()
TrackAssociatorByPosDeltaR.ComponentName = 'TrackAssociatorByDeltaR'
TrackAssociatorByPosDeltaR.method = cms.string('momdr')
TrackAssociatorByPosDeltaR.QCut = cms.double(0.5)
TrackAssociatorByPosDeltaR.ConsiderAllSimHits = cms.bool(True)

#
# Configuration for Muon track extractor
#

selectedVertices = cms.EDFilter("VertexSelector",
    src = cms.InputTag('offlinePrimaryVertices'),
    cut = cms.string("isValid & ndof >= 4 & chi2 > 0 & tracksSize > 0 & abs(z) < 24 & abs(position.Rho) < 2."),
    filter = cms.bool(False)
)

selectedFirstPrimaryVertex = cms.EDFilter("PATSingleVertexSelector",
    mode = cms.string('firstVertex'),
    vertices = cms.InputTag('selectedVertices'),
    filter = cms.bool(False)
)

muonPt5 = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("pt > 5."),
    filter = cms.bool(False)
)

muonPt10 = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("pt > 10."),
    filter = cms.bool(False)
)

muonPt20 = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("pt > 20."),
    filter = cms.bool(False)
)

import PhysicsTools.RecoAlgos.recoTrackSelector_cfi
staMuonsPt10 = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
staMuonsPt10.ptMin = cms.double(10.0)
staMuonsPt10.quality = cms.vstring('')
staMuonsPt10.minHit = cms.int32(0)
staMuonsPt10.src = cms.InputTag("standAloneMuons:UpdatedAtVtx")

staMuonsPt20 = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
staMuonsPt20.ptMin = cms.double(20.0)
staMuonsPt20.quality = cms.vstring('')
staMuonsPt20.minHit = cms.int32(0)
staMuonsPt20.src = cms.InputTag("standAloneMuons:UpdatedAtVtx")

#-----------------------------------------------------------------------------------------------------------------------

bestMuonLoose = cms.EDProducer("MuonTrackProducer",
   muonsTag = cms.InputTag("muons"),
   inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
   inputCSCSegmentCollection = cms.InputTag("cscSegments"),
   vxtTag = cms.InputTag("selectedVertices"),
   selectionTags = cms.vstring('AllGlobalMuons'),
   trackType = cms.string('globalTrackLoose')
)

bestMuonLoose5 = cms.EDProducer("MuonTrackProducer",
   muonsTag = cms.InputTag("muonPt5"),
   inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
   inputCSCSegmentCollection = cms.InputTag("cscSegments"),
   vxtTag = cms.InputTag("selectedVertices"),
   selectionTags = cms.vstring('AllGlobalMuons'),
   trackType = cms.string('globalTrackLoose')
)

bestMuonLoose10 = cms.EDProducer("MuonTrackProducer",
   muonsTag = cms.InputTag("muonPt10"),
   inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
   inputCSCSegmentCollection = cms.InputTag("cscSegments"),
   vxtTag = cms.InputTag("selectedVertices"),
   selectionTags = cms.vstring('AllGlobalMuons'),
   trackType = cms.string('globalTrackLoose')
)

bestMuonLoose20 = cms.EDProducer("MuonTrackProducer",
   muonsTag = cms.InputTag("muonPt20"),
   inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
   inputCSCSegmentCollection = cms.InputTag("cscSegments"),
   vxtTag = cms.InputTag("selectedVertices"),
   selectionTags = cms.vstring('AllGlobalMuons'),
   trackType = cms.string('globalTrackLoose')
)

#-----------------------------------------------------------------------------------------------------------------------

bestMuonLoose2 = cms.EDProducer("MuonTrackProducer",
   muonsTag = cms.InputTag("muons"),
   inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
   inputCSCSegmentCollection = cms.InputTag("cscSegments"),
   vxtTag = cms.InputTag("selectedVertices"),
   selectionTags = cms.vstring('AllGlobalMuons'),
   trackType = cms.string('globalTrackLoose2')
)

bestMuonLoose52 = cms.EDProducer("MuonTrackProducer",
   muonsTag = cms.InputTag("muonPt5"),
   inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
   inputCSCSegmentCollection = cms.InputTag("cscSegments"),
   vxtTag = cms.InputTag("selectedVertices"),
   selectionTags = cms.vstring('AllGlobalMuons'),
   trackType = cms.string('globalTrackLoose2')
)

bestMuonLoose102 = cms.EDProducer("MuonTrackProducer",
   muonsTag = cms.InputTag("muonPt10"),
   inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
   inputCSCSegmentCollection = cms.InputTag("cscSegments"),
   vxtTag = cms.InputTag("selectedVertices"),
   selectionTags = cms.vstring('AllGlobalMuons'),
   trackType = cms.string('globalTrackLoose2')
)

bestMuonLoose202 = cms.EDProducer("MuonTrackProducer",
   muonsTag = cms.InputTag("muonPt20"),
   inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
   inputCSCSegmentCollection = cms.InputTag("cscSegments"),
   vxtTag = cms.InputTag("selectedVertices"),
   selectionTags = cms.vstring('AllGlobalMuons'),
   trackType = cms.string('globalTrackLoose2')
)

#-----------------------------------------------------------------------------------------------------------------------

bestMuonTight = cms.EDProducer("MuonTrackProducer",
   muonsTag = cms.InputTag("muons"),
   inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
   inputCSCSegmentCollection = cms.InputTag("cscSegments"),
   vxtTag = cms.InputTag("selectedVertices"),
   selectionTags = cms.vstring('AllGlobalMuons'),
   trackType = cms.string('globalTrackTight')
)

bestMuonTight5 = cms.EDProducer("MuonTrackProducer",
   muonsTag = cms.InputTag("muonPt5"),
   inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
   inputCSCSegmentCollection = cms.InputTag("cscSegments"),
   vxtTag = cms.InputTag("selectedVertices"),
   selectionTags = cms.vstring('AllGlobalMuons'),
   trackType = cms.string('globalTrackTight')
)

bestMuonTight10 = cms.EDProducer("MuonTrackProducer",
   muonsTag = cms.InputTag("muonPt10"),
   inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
   inputCSCSegmentCollection = cms.InputTag("cscSegments"),
   vxtTag = cms.InputTag("selectedVertices"),
   selectionTags = cms.vstring('AllGlobalMuons'),
   trackType = cms.string('globalTrackTight')
)

bestMuonTight20 = cms.EDProducer("MuonTrackProducer",
   muonsTag = cms.InputTag("muonPt20"),
   inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
   inputCSCSegmentCollection = cms.InputTag("cscSegments"),
   vxtTag = cms.InputTag("selectedVertices"),
   selectionTags = cms.vstring('AllGlobalMuons'),
   trackType = cms.string('globalTrackTight')
)

#-----------------------------------------------------------------------------------------------------------------------

muonColl_seq = cms.Sequence( staMuonsPt10 *
			     staMuonsPt20 
			     #muonPt5 * 
			     #muonPt10 *
			     #muonPt20
  			   )

bestMuon_seq = cms.Sequence( bestMuonLoose #* bestMuonLoose5 *  bestMuonLoose10 * bestMuonLoose20 *
			     *bestMuonTight #* bestMuonTight5 * bestMuonTight10 * bestMuonTight20 *
			     #bestMuonLoose2 * bestMuonLoose52  * bestMuonLoose102  * bestMuonLoose202 
			   )

##############################################################################################################################################

import SimMuon.MCTruth.MuonTrackProducer_cfi
extractedGlobalMuons = SimMuon.MCTruth.MuonTrackProducer_cfi.muonTrackProducer.clone()
extractedGlobalMuons.selectionTags = ('AllGlobalMuons',)
extractedGlobalMuons.trackType = "globalTrack"

extractedSTAMuons = SimMuon.MCTruth.MuonTrackProducer_cfi.muonTrackProducer.clone()
extractedSTAMuons.selectionTags = ('AllGlobalMuons',)
extractedSTAMuons.trackType = "outerTrack"
extractedMuonTracks_seq = cms.Sequence( extractedGlobalMuons * extractedSTAMuons )

#
# Configuration for Seed track extractor
#
#
#import SimMuon.MCTruth.SeedToTrackProducer_cfi
#seedsOfSTAmuons = SimMuon.MCTruth.SeedToTrackProducer_cfi.SeedToTrackProducer.clone()
#seedsOfSTAmuons.L2seedsCollection = cms.InputTag("ancientMuonSeed")
#seedsOfSTAmuons_seq = cms.Sequence( seedsOfSTAmuons )

# select probe tracks
#import PhysicsTools.RecoAlgos.recoTrackSelector_cfi
probeTracks = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
probeTracks.quality = cms.vstring('highPurity')
probeTracks.tip = cms.double(3.5)
probeTracks.lip = cms.double(30.)
probeTracks.ptMin = cms.double(4.0)
probeTracks.minRapidity = cms.double(-2.4)
probeTracks.maxRapidity = cms.double(2.4)
probeTracks_seq = cms.Sequence( probeTracks )

#
# Associators for Full Sim + Reco:
#

tpToTkmuTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByHits'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
#    label_tr = cms.InputTag('generalTracks')
    label_tr = cms.InputTag('probeTracks')
)

tpToStaTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('standAloneMuons','')
)

tpToStaUpdTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('standAloneMuons','UpdatedAtVtx')
)

tpToGlbTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('extractedGlobalMuons')
)

tpToStaSETTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('standAloneSETMuons','')
)

tpToStaSETUpdTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('standAloneSETMuons','UpdatedAtVtx')
)

tpToGlbSETTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('globalSETMuons')
)

tpToTevFirstTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('tevMuons','firstHit')
)

tpToTevPickyTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('tevMuons','picky')
)
tpToTevDytTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('tevMuons','dyt')
)

tpToL2TrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('hltL2Muons','')
)

tpToL2UpdTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('hltL2Muons','UpdatedAtVtx')
)

tpToL3TrackAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('hltL3Muons')
)

tpToL3TkTrackTrackAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('OnlineTrackAssociatorByHits'),
    label_tp = cms.InputTag('mix','MergedTrackTruth'),
    label_tr = cms.InputTag('hltL3TkTracksFromL2','')
)

tpToL3L2TrackTrackAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('OnlineTrackAssociatorByHits'),
    label_tp = cms.InputTag('mix','MergedTrackTruth'),
    label_tr = cms.InputTag('hltL3Muons:L2Seeded')
)



#MuonAssociation
import SimMuon.MCTruth.MuonAssociatorByHits_cfi

tpToTkMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
#tpToStaSeedAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaUpdMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToGlbMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()

tpToLooseMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToLoose5MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToLoose10MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToLoose20MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()

tpToLoose2MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToLoose52MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToLoose102MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToLoose202MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()

tpToTightMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToTight5MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToTight10MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToTight20MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()

tpToStaRefitMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaRefitUpdMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaSETMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaSETUpdMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToGlbSETMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToTevFirstMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToTevPickyMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToTevDytMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToL3TkMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToL2MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToL2UpdMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToL3MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()

# few more association modules usable for the Upgrade TP studies 
tpToTkSelMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaUpdSelMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaUpdSel2MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()

tpToStaUpd10SelMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaUpd20SelMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()

tpToGlbSelMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToGlbSel2MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToGlbSel3MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToGlbSel4MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()

tpToLooseSelMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToLooseSel2MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToLooseSel3MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()

tpToTightSelMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToTightSel2MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToTightSel3MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()

tpToTkMuonAssociation.tpTag = 'mix:MergedTrackTruth'
#tpToTkMuonAssociation.tracksTag = 'generalTracks'
tpToTkMuonAssociation.tracksTag = 'probeTracks'
tpToTkMuonAssociation.UseTracker = True
tpToTkMuonAssociation.UseMuon = False

#tpToStaSeedAssociation.tpTag = 'mix:MergedTrackTruth'
#tpToStaSeedAssociation.tracksTag = 'seedsOfSTAmuons'
#tpToStaSeedAssociation.UseTracker = False
#tpToStaSeedAssociation.UseMuon = True
#

tpToStaMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaMuonAssociation.tracksTag = 'standAloneMuons'
tpToStaMuonAssociation.UseTracker = False
tpToStaMuonAssociation.UseMuon = True

tpToStaUpdMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaUpdMuonAssociation.tracksTag = 'standAloneMuons:UpdatedAtVtx'
tpToStaUpdMuonAssociation.UseTracker = False
tpToStaUpdMuonAssociation.UseMuon = True

tpToGlbMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToGlbMuonAssociation.tracksTag = 'extractedGlobalMuons'
tpToGlbMuonAssociation.UseTracker = True
tpToGlbMuonAssociation.UseMuon = True

#######################################################################

tpToLooseMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToLooseMuonAssociation.tracksTag = 'bestMuonLoose'
tpToLooseMuonAssociation.UseTracker = True
tpToLooseMuonAssociation.UseMuon = True

tpToLoose5MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToLoose5MuonAssociation.tracksTag = 'bestMuonLoose5'
tpToLoose5MuonAssociation.UseTracker = True
tpToLoose5MuonAssociation.UseMuon = True

tpToLoose10MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToLoose10MuonAssociation.tracksTag = 'bestMuonLoose10'
tpToLoose10MuonAssociation.UseTracker = True
tpToLoose10MuonAssociation.UseMuon = True

tpToLoose20MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToLoose20MuonAssociation.tracksTag = 'bestMuonLoose20'
tpToLoose20MuonAssociation.UseTracker = True
tpToLoose20MuonAssociation.UseMuon = True

tpToLoose2MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToLoose2MuonAssociation.tracksTag = 'bestMuonLoose2'
tpToLoose2MuonAssociation.UseTracker = True
tpToLoose2MuonAssociation.UseMuon = True

tpToLoose52MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToLoose52MuonAssociation.tracksTag = 'bestMuonLoose52'
tpToLoose52MuonAssociation.UseTracker = True
tpToLoose52MuonAssociation.UseMuon = True

tpToLoose102MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToLoose102MuonAssociation.tracksTag = 'bestMuonLoose102'
tpToLoose102MuonAssociation.UseTracker = True
tpToLoose102MuonAssociation.UseMuon = True

tpToLoose202MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToLoose202MuonAssociation.tracksTag = 'bestMuonLoose202'
tpToLoose202MuonAssociation.UseTracker = True
tpToLoose202MuonAssociation.UseMuon = True

tpToTightMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToTightMuonAssociation.tracksTag = 'bestMuonTight'
tpToTightMuonAssociation.UseTracker = True
tpToTightMuonAssociation.UseMuon = True

tpToTight5MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToTight5MuonAssociation.tracksTag = 'bestMuonTight5'
tpToTight5MuonAssociation.UseTracker = True
tpToTight5MuonAssociation.UseMuon = True

tpToTight10MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToTight10MuonAssociation.tracksTag = 'bestMuonTight10'
tpToTight10MuonAssociation.UseTracker = True
tpToTight10MuonAssociation.UseMuon = True

tpToTight20MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToTight20MuonAssociation.tracksTag = 'bestMuonTight20'
tpToTight20MuonAssociation.UseTracker = True
tpToTight20MuonAssociation.UseMuon = True

#######################################################################

tpToStaRefitMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaRefitMuonAssociation.tracksTag = 'refittedStandAloneMuons'
tpToStaRefitMuonAssociation.UseTracker = False
tpToStaRefitMuonAssociation.UseMuon = True

tpToStaRefitUpdMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaRefitUpdMuonAssociation.tracksTag = 'refittedStandAloneMuons:UpdatedAtVtx'
tpToStaRefitUpdMuonAssociation.UseTracker = False
tpToStaRefitUpdMuonAssociation.UseMuon = True

tpToStaSETMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaSETMuonAssociation.tracksTag = 'standAloneSETMuons'
tpToStaSETMuonAssociation.UseTracker = False
tpToStaSETMuonAssociation.UseMuon = True

tpToStaSETUpdMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaSETUpdMuonAssociation.tracksTag = 'standAloneSETMuons:UpdatedAtVtx'
tpToStaSETUpdMuonAssociation.UseTracker = False
tpToStaSETUpdMuonAssociation.UseMuon = True

tpToGlbSETMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToGlbSETMuonAssociation.tracksTag = 'globalSETMuons'
tpToGlbSETMuonAssociation.UseTracker = True
tpToGlbSETMuonAssociation.UseMuon = True

tpToTevFirstMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToTevFirstMuonAssociation.tracksTag = 'tevMuons:firstHit'
tpToTevFirstMuonAssociation.UseTracker = True
tpToTevFirstMuonAssociation.UseMuon = True

tpToTevPickyMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToTevPickyMuonAssociation.tracksTag = 'tevMuons:picky'
tpToTevPickyMuonAssociation.UseTracker = True
tpToTevPickyMuonAssociation.UseMuon = True

tpToTevDytMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToTevDytMuonAssociation.tracksTag = 'tevMuons:dyt'
tpToTevDytMuonAssociation.UseTracker = True
tpToTevDytMuonAssociation.UseMuon = True

tpToL3TkMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToL3TkMuonAssociation.tracksTag = 'hltL3TkTracksFromL2'
tpToL3TkMuonAssociation.DTrechitTag = 'hltDt1DRecHits'
tpToL3TkMuonAssociation.UseTracker = True
tpToL3TkMuonAssociation.UseMuon = False
tpToL3TkMuonAssociation.ignoreMissingTrackCollection = True
tpToL3TkMuonAssociation.UseSplitting = False
tpToL3TkMuonAssociation.UseGrouped = False

tpToL2MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToL2MuonAssociation.tracksTag = 'hltL2Muons'
tpToL2MuonAssociation.DTrechitTag = 'hltDt1DRecHits'
tpToL2MuonAssociation.UseTracker = False
tpToL2MuonAssociation.UseMuon = True
tpToL2MuonAssociation.ignoreMissingTrackCollection = True

tpToL2UpdMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToL2UpdMuonAssociation.tracksTag = 'hltL2Muons:UpdatedAtVtx'
tpToL2UpdMuonAssociation.DTrechitTag = 'hltDt1DRecHits'
tpToL2UpdMuonAssociation.UseTracker = False
tpToL2UpdMuonAssociation.UseMuon = True
tpToL2UpdMuonAssociation.ignoreMissingTrackCollection = True

tpToL3MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToL3MuonAssociation.tracksTag = 'hltL3Muons'
tpToL3MuonAssociation.DTrechitTag = 'hltDt1DRecHits'
tpToL3MuonAssociation.UseTracker = True
tpToL3MuonAssociation.UseMuon = True
tpToL3MuonAssociation.ignoreMissingTrackCollection = True
tpToL3MuonAssociation.UseSplitting = False
tpToL3MuonAssociation.UseGrouped = False

# few more association modules usable for the Upgrade TP studies 
#tpToTkSelMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
#tpToStaUpdSelMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
#tpToGlbSelMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()

# about STA

tpToTkSelMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToTkSelMuonAssociation.tracksTag = 'probeTracks'
tpToTkSelMuonAssociation.UseTracker = True
tpToTkSelMuonAssociation.UseMuon = False
tpToTkSelMuonAssociation.EfficiencyCut_track = cms.double(0.5)
tpToTkSelMuonAssociation.PurityCut_track = cms.double(0.75)

tpToStaUpdSelMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaUpdSelMuonAssociation.tracksTag = 'standAloneMuons:UpdatedAtVtx' 
tpToStaUpdSelMuonAssociation.UseTracker = False
tpToStaUpdSelMuonAssociation.UseMuon = True
tpToStaUpdSelMuonAssociation.includeZeroHitMuons = False

tpToStaUpdSel2MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaUpdSel2MuonAssociation.tracksTag = 'standAloneMuons:UpdatedAtVtx' 
tpToStaUpdSel2MuonAssociation.UseTracker = False
tpToStaUpdSel2MuonAssociation.UseMuon = True
tpToStaUpdSel2MuonAssociation.includeZeroHitMuons = False
tpToStaUpdSel2MuonAssociation.PurityCut_muon = cms.double(0.75)

tpToStaUpd10SelMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaUpd10SelMuonAssociation.tracksTag = 'staMuonsPt10'
tpToStaUpd10SelMuonAssociation.UseTracker = False
tpToStaUpd10SelMuonAssociation.UseMuon = True
tpToStaUpd10SelMuonAssociation.includeZeroHitMuons = False

tpToStaUpd20SelMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaUpd20SelMuonAssociation.tracksTag = 'staMuonsPt20'
tpToStaUpd20SelMuonAssociation.UseTracker = False
tpToStaUpd20SelMuonAssociation.UseMuon = True
tpToStaUpd20SelMuonAssociation.includeZeroHitMuons = False

# about global

tpToGlbSelMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToGlbSelMuonAssociation.tracksTag = 'extractedGlobalMuons' 
tpToGlbSelMuonAssociation.UseTracker = True
tpToGlbSelMuonAssociation.UseMuon = True
tpToGlbSelMuonAssociation.EfficiencyCut_track = cms.double(0.5)
tpToGlbSelMuonAssociation.PurityCut_track = cms.double(0.75)
tpToGlbSelMuonAssociation.acceptOneStubMatchings = False
tpToGlbSelMuonAssociation.includeZeroHitMuons = False

tpToGlbSel2MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToGlbSel2MuonAssociation.tracksTag = 'extractedGlobalMuons' 
tpToGlbSel2MuonAssociation.UseTracker = True
tpToGlbSel2MuonAssociation.UseMuon = True
tpToGlbSel2MuonAssociation.EfficiencyCut_track = cms.double(0.5)
tpToGlbSel2MuonAssociation.PurityCut_track = cms.double(0.75)
#tpToGlbSel2MuonAssociation.acceptOneStubMatchings = False
tpToGlbSel2MuonAssociation.includeZeroHitMuons = False

tpToGlbSel3MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToGlbSel3MuonAssociation.tracksTag = 'extractedGlobalMuons' 
tpToGlbSel3MuonAssociation.UseTracker = True
tpToGlbSel3MuonAssociation.UseMuon = True
tpToGlbSel3MuonAssociation.EfficiencyCut_track = cms.double(0.5)
tpToGlbSel3MuonAssociation.PurityCut_track = cms.double(0.75)
tpToGlbSel3MuonAssociation.PurityCut_muon = cms.double(0.75)
tpToGlbSel3MuonAssociation.acceptOneStubMatchings = False
tpToGlbSel3MuonAssociation.includeZeroHitMuons = False

tpToGlbSel4MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToGlbSel4MuonAssociation.tracksTag = 'extractedGlobalMuons' 
tpToGlbSel4MuonAssociation.UseTracker = True
tpToGlbSel4MuonAssociation.UseMuon = True
tpToGlbSel4MuonAssociation.EfficiencyCut_track = cms.double(0.5)
tpToGlbSel4MuonAssociation.PurityCut_track = cms.double(0.75)
tpToGlbSel4MuonAssociation.PurityCut_muon = cms.double(0.75)
#tpToGlbSel4MuonAssociation.acceptOneStubMatchings = False
tpToGlbSel4MuonAssociation.includeZeroHitMuons = False

#####################################################################################################

tpToLooseSelMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToLooseSelMuonAssociation.tracksTag = 'bestMuonLoose' 
tpToLooseSelMuonAssociation.UseTracker = True
tpToLooseSelMuonAssociation.UseMuon = True
tpToLooseSelMuonAssociation.EfficiencyCut_track = cms.double(0.5)
tpToLooseSelMuonAssociation.PurityCut_track = cms.double(0.75)
tpToLooseSelMuonAssociation.acceptOneStubMatchings = False
tpToLooseSelMuonAssociation.includeZeroHitMuons = False

tpToLooseSel3MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToLooseSel3MuonAssociation.tracksTag = 'bestMuonLoose' 
tpToLooseSel3MuonAssociation.UseTracker = True
tpToLooseSel3MuonAssociation.UseMuon = True
tpToLooseSel3MuonAssociation.EfficiencyCut_track = cms.double(0.5)
tpToLooseSel3MuonAssociation.PurityCut_track = cms.double(0.75)
tpToLooseSel3MuonAssociation.PurityCut_muon = cms.double(0.75)
tpToLooseSel3MuonAssociation.acceptOneStubMatchings = False
tpToLooseSel3MuonAssociation.includeZeroHitMuons = False

tpToLooseSel2MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToLooseSel2MuonAssociation.tracksTag = 'bestMuonLoose' 
tpToLooseSel2MuonAssociation.UseTracker = True
tpToLooseSel2MuonAssociation.UseMuon = True
tpToLooseSel2MuonAssociation.EfficiencyCut_track = cms.double(0.5)
tpToLooseSel2MuonAssociation.PurityCut_track = cms.double(0.75)
#tpToLooseSel2MuonAssociation.acceptOneStubMatchings = False
tpToLooseSel2MuonAssociation.includeZeroHitMuons = False

tpToTightSelMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToTightSelMuonAssociation.tracksTag = 'bestMuonTight' 
tpToTightSelMuonAssociation.UseTracker = True
tpToTightSelMuonAssociation.UseMuon = True
tpToTightSelMuonAssociation.EfficiencyCut_track = cms.double(0.5)
tpToTightSelMuonAssociation.PurityCut_track = cms.double(0.75)
tpToTightSelMuonAssociation.acceptOneStubMatchings = False
tpToTightSelMuonAssociation.includeZeroHitMuons = False

tpToTightSel3MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToTightSel3MuonAssociation.tracksTag = 'bestMuonTight' 
tpToTightSel3MuonAssociation.UseTracker = True
tpToTightSel3MuonAssociation.UseMuon = True
tpToTightSel3MuonAssociation.EfficiencyCut_track = cms.double(0.5)
tpToTightSel3MuonAssociation.PurityCut_track = cms.double(0.75)
tpToTightSel3MuonAssociation.PurityCut_muon = cms.double(0.75)
tpToTightSel3MuonAssociation.acceptOneStubMatchings = False
tpToTightSel3MuonAssociation.includeZeroHitMuons = False

tpToTightSel2MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToTightSel2MuonAssociation.tracksTag = 'bestMuonTight' 
tpToTightSel2MuonAssociation.UseTracker = True
tpToTightSel2MuonAssociation.UseMuon = True
tpToTightSel2MuonAssociation.EfficiencyCut_track = cms.double(0.5)
tpToTightSel2MuonAssociation.PurityCut_track = cms.double(0.75)
#tpToTightSel2MuonAssociation.acceptOneStubMatchings = False
tpToTightSel2MuonAssociation.includeZeroHitMuons = False
 
#
# Associators for cosmics:
#

tpToTkCosmicTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByHits'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('ctfWithMaterialTracksP5LHCNavigation')
)

tpToStaCosmicTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('cosmicMuons')
)

tpToGlbCosmicTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('globalCosmicMuons')
)

tpToTkCosmicMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaCosmicMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToGlbCosmicMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()

tpToTkCosmicMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToTkCosmicMuonAssociation.tracksTag = 'ctfWithMaterialTracksP5LHCNavigation'
tpToTkCosmicMuonAssociation.UseTracker = True
tpToTkCosmicMuonAssociation.UseMuon = False

tpToStaCosmicMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaCosmicMuonAssociation.tracksTag = 'cosmicMuons'
tpToStaCosmicMuonAssociation.UseTracker = False
tpToStaCosmicMuonAssociation.UseMuon = True

tpToGlbCosmicMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToGlbCosmicMuonAssociation.tracksTag = 'globalCosmicMuons'
tpToGlbCosmicMuonAssociation.UseTracker = True
tpToGlbCosmicMuonAssociation.UseMuon = True


#
# The full-sim association sequences
#

muonAssociation_seq = cms.Sequence(
    selectedVertices
    + extractedMuonTracks_seq
    + muonColl_seq
    + bestMuon_seq
#    +seedsOfSTAmuons_seq
    +probeTracks_seq+(tpToTkMuonAssociation+tpToTkmuTrackAssociation)
#    +(tpToStaSeedAssociation+tpToStaMuonAssociation+tpToStaUpdMuonAssociation+tpToGlbMuonAssociation)
    +(tpToStaMuonAssociation+tpToStaUpdMuonAssociation+tpToGlbMuonAssociation
    +tpToLooseMuonAssociation#+tpToLoose5MuonAssociation+tpToLoose10MuonAssociation+tpToLoose20MuonAssociation
#    +tpToLoose2MuonAssociation+tpToLoose52MuonAssociation+tpToLoose102MuonAssociation+tpToLoose202MuonAssociation
    +tpToTightMuonAssociation#+tpToTight5MuonAssociation+tpToTight10MuonAssociation+tpToTight20MuonAssociation
    )
#   +(tpToStaTrackAssociation+tpToStaUpdTrackAssociation+tpToGlbTrackAssociation)
#    
# few more association modules usable for the Upgrade TP studies 
    +(tpToTkSelMuonAssociation+tpToStaUpdSelMuonAssociation+tpToStaUpdSel2MuonAssociation+tpToGlbSelMuonAssociation+tpToGlbSel2MuonAssociation+tpToGlbSel3MuonAssociation+tpToGlbSel4MuonAssociation+tpToStaUpd10SelMuonAssociation+tpToStaUpd20SelMuonAssociation)
    +(tpToLooseSelMuonAssociation+tpToLooseSel2MuonAssociation+tpToLooseSel3MuonAssociation)
    +(tpToTightSelMuonAssociation+tpToTightSel2MuonAssociation+tpToTightSel3MuonAssociation) 
)
muonAssociationTEV_seq = cms.Sequence(
    (tpToTevFirstMuonAssociation+tpToTevPickyMuonAssociation+tpToTevDytMuonAssociation)
#    +(tpToTevFirstTrackAssociation+tpToTevPickyTrackAssociation)
)
muonAssociationRefit_seq = cms.Sequence(
    (tpToStaRefitMuonAssociation+tpToStaRefitUpdMuonAssociation)
)
muonAssociationSET_seq = cms.Sequence(
    (tpToStaSETMuonAssociation+tpToStaSETUpdMuonAssociation+tpToGlbSETMuonAssociation)
#    +(tpToStaSETTrackAssociation+tpToStaSETUpdTrackAssociation+tpToGlbSETTrackAssociation)
)
muonAssociationCosmic_seq = cms.Sequence(
    (tpToTkCosmicMuonAssociation+tpToStaCosmicMuonAssociation+tpToGlbCosmicMuonAssociation)
#    +(tpToTkCosmicTrackAssociation+tpToStaCosmicTrackAssociation+tpToGlbCosmicTrackAssociation)
)
muonAssociationHLT_seq = cms.Sequence(
    (tpToL2MuonAssociation+tpToL2UpdMuonAssociation+tpToL3MuonAssociation+tpToL3TkMuonAssociation)
#    +(tpToL2TrackAssociation+tpToL2UpdTrackAssociation+tpToL3TrackAssociation+tpToL3TkTrackTrackAssociation)
)


#
# Associators for Fast Sim
#

tpToTkmuTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByHits'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
#    label_tr = cms.InputTag('generalTracks')
    label_tr = cms.InputTag('probeTracks')
)

tpToStaTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('standAloneMuons','')
)

tpToStaUpdTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('standAloneMuons','UpdatedAtVtx')
)

tpToGlbTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('extractedGlobalMuons')
)

tpToTevFirstTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('tevMuons','firstHit')
)

tpToTevPickyTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('tevMuons','picky')
)

tpToTevDytTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('tevMuons','dyt')
)

tpToL2TrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('hltL2Muons','')
)

tpToL2UpdTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('hltL2Muons','UpdatedAtVtx')
)

tpToL3TrackAssociationFS = cms.EDProducer("TrackAssociatorEDProducer",
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('hltL3Muons')
)

tpToL3TkTrackTrackAssociationFS = cms.EDProducer("TrackAssociatorEDProducer",
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('OnlineTrackAssociatorByHits'),
    label_tp = cms.InputTag('mix','MergedTrackTruth'),
    label_tr = cms.InputTag('hltL3TkTracksFromL2','')
)

tpToL3L2TrackTrackAssociationFS = cms.EDProducer("TrackAssociatorEDProducer",
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('OnlineTrackAssociatorByHits'),
    label_tp = cms.InputTag('mix','MergedTrackTruth'),
    label_tr = cms.InputTag('hltL3Muons:L2Seeded')
)


#MuonAssociation
import SimMuon.MCTruth.MuonAssociatorByHits_cfi

baseMuonAssociatorFS = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
baseMuonAssociatorFS.tpTag = 'mix:MergedTrackTruth'
baseMuonAssociatorFS.UseTracker = True
baseMuonAssociatorFS.UseMuon = True
baseMuonAssociatorFS.simtracksTag = "famosSimHits"
baseMuonAssociatorFS.DTsimhitsTag  = "MuonSimHits:MuonDTHits"
baseMuonAssociatorFS.CSCsimHitsTag = "MuonSimHits:MuonCSCHits"
baseMuonAssociatorFS.RPCsimhitsTag = "MuonSimHits:MuonRPCHits"
baseMuonAssociatorFS.GEMsimhitsTag = "MuonSimHits:MuonGEMHits"
baseMuonAssociatorFS.simtracksXFTag = "mix:famosSimHits"
baseMuonAssociatorFS.DTsimhitsXFTag  = "mix:MuonSimHitsMuonDTHits"
baseMuonAssociatorFS.CSCsimHitsXFTag = "mix:MuonSimHitsMuonCSCHits"
baseMuonAssociatorFS.RPCsimhitsXFTag = "mix:MuonSimHitsMuonRPCHits"
baseMuonAssociatorFS.GEMsimhitsXFTag = "mix:MuonSimHitsMuonGEMHits"
baseMuonAssociatorFS.ROUList = ['famosSimHitsTrackerHits']


tpToTkMuonAssociationFS  = baseMuonAssociatorFS.clone()
#tpToStaSeedAssociationFS = baseMuonAssociatorFS.clone()
tpToStaMuonAssociationFS  = baseMuonAssociatorFS.clone()
tpToStaUpdMuonAssociationFS  = baseMuonAssociatorFS.clone()
tpToStaRefitMuonAssociationFS  = baseMuonAssociatorFS.clone()
tpToStaRefitUpdMuonAssociationFS  = baseMuonAssociatorFS.clone()
tpToGlbMuonAssociationFS  = baseMuonAssociatorFS.clone()
tpToTevFirstMuonAssociationFS = baseMuonAssociatorFS.clone()
tpToTevPickyMuonAssociationFS = baseMuonAssociatorFS.clone()
tpToTevDytMuonAssociationFS = baseMuonAssociatorFS.clone()
tpToL3TkMuonAssociationFS = baseMuonAssociatorFS.clone()
tpToL2MuonAssociationFS   = baseMuonAssociatorFS.clone()
tpToL2UpdMuonAssociationFS   = baseMuonAssociatorFS.clone()
tpToL3MuonAssociationFS   = baseMuonAssociatorFS.clone()

#tpToTkMuonAssociationFS.tracksTag = 'generalTracks'
tpToTkMuonAssociationFS.tracksTag = 'probeTracks'
tpToTkMuonAssociationFS.tpTag = 'mix:MergedTrackTruth'
tpToTkMuonAssociationFS.UseTracker = True
tpToTkMuonAssociationFS.UseMuon = False

#tpToStaSeedAssociationFS.tpTag = 'mix:MergedTrackTruth'
#tpToStaSeedAssociationFS.tracksTag = 'seedsOfSTAmuons'
#tpToStaSeedAssociationFS.UseTracker = False
#tpToStaSeedAssociationFS.UseMuon = True

tpToStaMuonAssociationFS.tpTag = 'mix:MergedTrackTruth'
tpToStaMuonAssociationFS.tracksTag = 'standAloneMuons'
tpToStaMuonAssociationFS.UseTracker = False
tpToStaMuonAssociationFS.UseMuon = True

tpToStaUpdMuonAssociationFS.tracksTag = 'standAloneMuons:UpdatedAtVtx'
tpToStaUpdMuonAssociationFS.UseTracker = False
tpToStaUpdMuonAssociationFS.UseMuon = True

tpToStaRefitMuonAssociationFS.tracksTag = 'refittedStandAloneMuons'
tpToStaRefitMuonAssociationFS.UseTracker = False
tpToStaRefitMuonAssociationFS.UseMuon = True

tpToStaRefitUpdMuonAssociationFS.tracksTag = 'refittedStandAloneMuons:UpdatedAtVtx'
tpToStaRefitUpdMuonAssociationFS.UseTracker = False
tpToStaRefitUpdMuonAssociationFS.UseMuon = True

tpToGlbMuonAssociationFS.tracksTag = 'extractedGlobalMuons'
tpToGlbMuonAssociationFS.tpTag = 'mix:MergedTrackTruth'
tpToGlbMuonAssociationFS.UseTracker = True
tpToGlbMuonAssociationFS.UseMuon = True

tpToTevFirstMuonAssociationFS.tracksTag = 'tevMuons:firstHit'
tpToTevFirstMuonAssociationFS.tpTag = 'mix:MergedTrackTruth'
tpToTevFirstMuonAssociationFS.UseTracker = True
tpToTevFirstMuonAssociationFS.UseMuon = True

tpToTevPickyMuonAssociationFS.tracksTag = 'tevMuons:picky'
tpToTevPickyMuonAssociationFS.tpTag = 'mix:MergedTrackTruth'
tpToTevPickyMuonAssociationFS.UseTracker = True
tpToTevPickyMuonAssociationFS.UseMuon = True

tpToTevDytMuonAssociationFS.tracksTag = 'tevMuons:dyt'
tpToTevDytMuonAssociationFS.tpTag = 'mix:MergedTrackTruth'
tpToTevDytMuonAssociationFS.UseTracker = True
tpToTevDytMuonAssociationFS.UseMuon = True

tpToL3TkMuonAssociationFS.tracksTag = 'hltL3TkTracksFromL2'
tpToL3TkMuonAssociationFS.tpTag = 'mix:MergedTrackTruth'
tpToL3TkMuonAssociationFS.UseTracker = True
tpToL3TkMuonAssociationFS.UseMuon = False
tpToL3TkMuonAssociationFS.ignoreMissingTrackCollection = True
tpToL3TkMuonAssociationFS.UseSplitting = False
tpToL3TkMuonAssociationFS.UseGrouped = False

tpToL2MuonAssociationFS.tracksTag = 'hltL2Muons'
tpToL2MuonAssociationFS.UseTracker = False
tpToL2MuonAssociationFS.UseMuon = True
tpToL2MuonAssociationFS.ignoreMissingTrackCollection = True

tpToL2UpdMuonAssociationFS.tracksTag = 'hltL2Muons:UpdatedAtVtx'
tpToL2UpdMuonAssociationFS.UseTracker = False
tpToL2UpdMuonAssociationFS.UseMuon = True
tpToL2UpdMuonAssociationFS.ignoreMissingTrackCollection = True

tpToL3MuonAssociationFS.tracksTag = 'hltL3Muons'
tpToL3MuonAssociationFS.tpTag = 'mix:MergedTrackTruth'
tpToL3MuonAssociationFS.UseTracker = True
tpToL3MuonAssociationFS.UseMuon = True
tpToL3MuonAssociationFS.ignoreMissingTrackCollection = True
tpToL3MuonAssociationFS.UseSplitting = False
tpToL3MuonAssociationFS.UseGrouped = False



muonAssociationFastSim_seq = cms.Sequence(
        extractedMuonTracks_seq
#        +seedsOfSTAmuons_seq
        +probeTracks+(tpToTkMuonAssociationFS+tpToTkmuTrackAssociationFS) 
#        +(tpToStaSeedAssociationFS+tpToStaMuonAssociationFS+tpToStaUpdMuonAssociationFS+tpToGlbMuonAssociationFS)
        +(tpToStaMuonAssociationFS+tpToStaUpdMuonAssociationFS+tpToGlbMuonAssociationFS)
        +(tpToStaRefitMuonAssociationFS+tpToStaRefitUpdMuonAssociationFS)
        +(tpToTevFirstMuonAssociationFS+tpToTevPickyMuonAssociationFS+tpToTevDytMuonAssociationFS)
#        +tpToStaTrackAssociationFS+tpToStaUpdTrackAssociationFS+tpToGlbTrackAssociationFS
#        +tpToTevFirstTrackAssociationFS+tpToTevPickyTrackAssociationFS
        )
muonAssociationHLTFastSim_seq = cms.Sequence(
    tpToL2MuonAssociationFS+tpToL2UpdMuonAssociationFS+tpToL3MuonAssociationFS+tpToL3TkMuonAssociationFS
#    +tpToL2TrackAssociationFS+tpToL2UpdTrackAssociationFS+tpToL3TrackAssociationFS+tpToL3TkTrackTrackAssociationFS
    )

