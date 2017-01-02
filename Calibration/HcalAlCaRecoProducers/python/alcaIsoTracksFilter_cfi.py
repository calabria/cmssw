import FWCore.ParameterSet.Config as cms

AlcaIsoTracksFilter = cms.EDFilter("AlCaIsoTracksFilter",
                                   TrackLabel        = cms.InputTag("generalTracks"),
                                   VertexLabel       = cms.InputTag("offlinePrimaryVertices"),
                                   BeamSpotLabel     = cms.InputTag("offlineBeamSpot"),
                                   EBRecHitLabel     = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                                   EERecHitLabel     = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
                                   HBHERecHitLabel   = cms.InputTag("hbhereco"),
                                   TriggerEventLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
                                   TriggerResultLabel= cms.InputTag("TriggerResults","","HLT"),
                                   Triggers          = cms.vstring(),
                                   TrackQuality      = cms.string("highPurity"),
                                   ProcessName       = cms.string("HLT"),
                                   MinTrackPt        = cms.double(10.0),
                                   SlopeTrackPt      = cms.double(0.16),
                                   MaxDxyPV          = cms.double(10.0),
                                   MaxDzPV           = cms.double(100.0),
                                   MaxChi2           = cms.double(5.0),
                                   MaxDpOverP        = cms.double(0.1),
                                   MinOuterHit       = cms.int32(4),
                                   MinLayerCrossed   = cms.int32(8),
                                   MaxInMiss         = cms.int32(2),
                                   MaxOutMiss        = cms.int32(2),
                                   ConeRadius        = cms.double(34.98),
                                   ConeRadiusMIP     = cms.double(14.0),
                                   MinimumTrackP     = cms.double(20.0),
                                   MaximumEcalEnergy = cms.double(2.0),
                                   IsolationEnergy   = cms.double(10.0),
)
