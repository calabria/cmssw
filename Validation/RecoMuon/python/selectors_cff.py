import FWCore.ParameterSet.Config as cms

# TrackingParticle (MC truth) selectors
muonTPSet = cms.PSet(
    src = cms.InputTag("mix", "MergedTrackTruth"),
    pdgId = cms.vint32(13, -13),
    tip = cms.double(3.5),
    lip = cms.double(30.0),
    minHit = cms.int32(0),
    ptMin = cms.double(0.9),
    minRapidity = cms.double(-2.5),
    maxRapidity = cms.double(2.5),
    signalOnly = cms.bool(True),
    stableOnly = cms.bool(False),
    chargedOnly = cms.bool(True)
)

muonGPSet = cms.PSet(
    lipGP = cms.double(30.0),
    chargedOnlyGP = cms.bool(True),
    pdgIdGP = cms.vint32(13, -13),
    minRapidityGP = cms.double(-2.5),
    ptMinGP = cms.double(0.9),
    maxRapidityGP = cms.double(2.5),
    tipGP = cms.double(3.5),
    statusGP = cms.int32(1)
)

#muonTP = cms.EDFilter("TrackingParticleSelector",
#    muonTPSet
#)

# RecoTrack selectors
#muonGlb = cms.EDFilter("RecoTrackSelector",
#    src = cms.InputTag("globalMuons"),
#    tip = cms.double(3.5),
#    lip = cms.double(30.0),
#    minHit = cms.int32(8),
#    maxChi2 = cms.double(999),
#    ptMin = cms.double(0.8),
#    quality = cms.string("Chi2"),
#    minRapidity = cms.double(-2.5),
#    maxRapidity = cms.double(2.5)
#)
#
#muonSta = cms.EDFilter("RecoTrackSelector",
#    src = cms.InputTag("standAloneMuons","UpdatedAtVtx"),
#    tip = cms.double(999.0),
#    lip = cms.double(999.0),
#    minHit = cms.int32(1),
#    maxChi2 = cms.double(999),
#    ptMin = cms.double(0.8),
#    quality = cms.string("Chi2"),
#    minRapidity = cms.double(-2.5),
#    maxRapidity = cms.double(2.5)
#)

#muonSelector_step = cms.Sequence(muonTP+muonGlb+muonSta)

#muonSelector_seq = cms.Sequence(muonTP)
