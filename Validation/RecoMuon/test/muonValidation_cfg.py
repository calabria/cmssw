import FWCore.ParameterSet.Config as cms

processName = "MuonSuite"
process = cms.Process(processName)

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
    #'file:/lustre/cms/store/group/upgrade/muon/RecoFolder/DYToMuMu_2019_3Step/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/calabria_DYToMuMu_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC23patch1_2019_3Step_OK/60353824f22fa9fdbd2b14ad88cccbca/step3_1000_1_YM5.root',
    #'file:/lustre/cms//store/group/upgrade/muon/RecoFolder/DS1000_2023_3Step_FullScope/DarkSUSY_MH-125_MGammaD-20000_ctau1000_14TeV_madgraph-pythia6-tauola/calabria_DS1000_GEN-SIM-RECO_CMSSW_6_2_0_SLHC26patch2_2023_3Step_FullScope/215357deaab3db6bc030de8be3791da2/step3_10_3_fA5.root',
    'file:/cmshome/calabria/ValidazioneOfficial2/CMSSW_6_2_0_SLHC26_patch2/src/L1Trigger/L1IntegratedMuonTrigger/test/CfgForDisplacedMuons/step3.root',
    ] );
secFiles.extend((
    #'file:/lustre/cms/store/group/upgrade/muon/RecoFolder/DYToMuMu_2019_2Step_2/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/calabria_DYToMuMu_GEN-SIM-DIGI-RAW_CMSSW_6_2_0_SLHC23patch1_2019Scenario_2Step_GEMSH/23d4646c3e8a6be200238397ea8208ad/step2_427_1_2nJ.root',
    '/store/mc/TP2023HGCALDR/DarkSUSY_MH-125_MGammaD-20000_ctau1000_14TeV_madgraph-pythia6-tauola/GEN-SIM-DIGI-RAW/HGCALForMUO_PU140BX25_newsplit_PH2_1K_FB_V6-v2/40000/0AC5C84E-503C-E511-9111-002590E3A0FC.root',
    '/store/mc/TP2023HGCALDR/DarkSUSY_MH-125_MGammaD-20000_ctau1000_14TeV_madgraph-pythia6-tauola/GEN-SIM-DIGI-RAW/HGCALForMUO_PU140BX25_newsplit_PH2_1K_FB_V6-v2/40000/10743163-503C-E511-A56A-002590E3A0FC.root',
    '/store/mc/TP2023HGCALDR/DarkSUSY_MH-125_MGammaD-20000_ctau1000_14TeV_madgraph-pythia6-tauola/GEN-SIM-DIGI-RAW/HGCALForMUO_PU140BX25_newsplit_PH2_1K_FB_V6-v2/40000/6C6DA846-503C-E511-9F4A-0CC47A13D16A.root',
    '/store/mc/TP2023HGCALDR/DarkSUSY_MH-125_MGammaD-20000_ctau1000_14TeV_madgraph-pythia6-tauola/GEN-SIM-DIGI-RAW/HGCALForMUO_PU140BX25_newsplit_PH2_1K_FB_V6-v2/40000/80E1F840-503C-E511-9487-002590E1E9B8.root',
    '/store/mc/TP2023HGCALDR/DarkSUSY_MH-125_MGammaD-20000_ctau1000_14TeV_madgraph-pythia6-tauola/GEN-SIM-DIGI-RAW/HGCALForMUO_PU140BX25_newsplit_PH2_1K_FB_V6-v2/40000/94446E1B-703C-E511-9A9B-90B11C2AB44B.root',
    '/store/mc/TP2023HGCALDR/DarkSUSY_MH-125_MGammaD-20000_ctau1000_14TeV_madgraph-pythia6-tauola/GEN-SIM-DIGI-RAW/HGCALForMUO_PU140BX25_newsplit_PH2_1K_FB_V6-v2/40000/C069B1C9-503C-E511-B2E3-002590E39F36.root',
    '/store/mc/TP2023HGCALDR/DarkSUSY_MH-125_MGammaD-20000_ctau1000_14TeV_madgraph-pythia6-tauola/GEN-SIM-DIGI-RAW/HGCALForMUO_PU140BX25_newsplit_PH2_1K_FB_V6-v2/40000/E2D2E3C5-503C-E511-B4DA-0CC47A13D216.root',
    '/store/mc/TP2023HGCALDR/DarkSUSY_MH-125_MGammaD-20000_ctau1000_14TeV_madgraph-pythia6-tauola/GEN-SIM-DIGI-RAW/HGCALForMUO_PU140BX25_newsplit_PH2_1K_FB_V6-v2/40000/EC30B357-503C-E511-A420-0CC47A13CEF4.root',
    ))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(50) )

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', "keep *_MEtoEDMConverter_*_"+processName),
    fileName = cms.untracked.string('validationEDM.root')
)
process.outpath = cms.EndPath(process.out)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 1


process.load('Configuration/StandardSequences/RawToDigi_cff')
process.raw2digi_step = cms.Path(process.RawToDigi)

process.load("Configuration/StandardSequences/SimulationRandomNumberGeneratorSeeds_cff")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.MEtoEDMConverter_step = cms.Path(process.MEtoEDMConverter)

process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023Muon_cff')
#process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
#process.load('Configuration.Geometry.GeometryExtended2019_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

#---- Validation stuffs ----#
## Default validation modules
process.load("Configuration.StandardSequences.Validation_cff")
process.validation_step = cms.Path(process.validation)
## Load muon validation modules
#process.recoMuonVMuAssoc.outputFileName = 'validationME.root'
process.muonValidation_step = cms.Path(process.recoMuonValidation)

process.load("SimMuon.MCTruth.MuonAssociatorByHits_cfi")
process.muonAssociatorByHitsCommonParameters.useGEMs = cms.bool(True)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.printTree = cms.EDAnalyzer("ParticleListDrawer",
    maxEventsToPrint = cms.untracked.int32(50),
    printVertex = cms.untracked.bool(False),
    printOnlyHardInteraction = cms.untracked.bool(False), # Print only status=3 particles. This will not work for Pythia8, which does not have any such particles.
    src = cms.InputTag("genParticles")
)
process.gen_step = cms.Path(process.printTree)

process.schedule = cms.Schedule(
    process.gen_step,
    #process.raw2digi_step,
    #process.validation_step,
    process.muonValidation_step,
    process.MEtoEDMConverter_step,process.outpath)

