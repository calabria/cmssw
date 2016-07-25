import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("FlatRandomPtAndD0GunProducer",
    PGunParameters = cms.PSet(
        MaxPt  = cms.double(100.00),
        MinPt  = cms.double(5.00),
        PartID = cms.vint32(-13),
        MaxEta = cms.double(3.0),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-3.0),
        MinPhi = cms.double(-3.14159265359),
        LxyMin = cms.double(100.0),#in cm
        LxyMax = cms.double(350.0),#in cm
        LzMin  = cms.double(-500.0),#in cm
        LzMax  = cms.double(500.0),#in cm
        dxyMin = cms.double(100.0),
        dxyMax = cms.double(200.0)
    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts
                           
    psethack = cms.string('single mu pt 100'),
    AddAntiParticle = cms.bool(False),
    firstRun = cms.untracked.uint32(1)
)