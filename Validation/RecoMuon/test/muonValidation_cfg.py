import FWCore.ParameterSet.Config as cms

processName = "MuonSuite"
process = cms.Process(processName)

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [

	'/store/group/upgrade/muon/RecoFolder/DYToMuMu_2019Age_3Step/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/calabria_DYToMuMu_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC23patch1_2019Age_3Step_OK7/401b8d05daa77b137fd55a5661197207/step3_1000_1_9NR.root',

 ] );


secFiles.extend( [
               ] )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', "keep *_MEtoEDMConverter_*_"+processName),
    fileName = cms.untracked.string('validationEDM.root')
)
process.outpath = cms.EndPath(process.out)

process.load('Configuration/StandardSequences/RawToDigi_cff')
process.raw2digi_step = cms.Path(process.RawToDigi)

process.load("Configuration/StandardSequences/SimulationRandomNumberGeneratorSeeds_cff")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.MEtoEDMConverter_step = cms.Path(process.MEtoEDMConverter)

process.load("Configuration.StandardSequences.Services_cff")
#process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')
#process.load('Configuration.Geometry.GeometryExtended2023Muon_cff')
process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2019_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'PH1_1K_FB_V2::All', '')

#---- Validation stuffs ----#
## Default validation modules
process.load("Configuration.StandardSequences.Validation_cff")
process.validation_step = cms.Path(process.validation)
## Load muon validation modules
#process.recoMuonVMuAssoc.outputFileName = 'validationME.root'
process.muonValidation_step = cms.Path(process.recoMuonValidation)

process.schedule = cms.Schedule(
    #process.raw2digi_step,
    #process.validation_step,
    process.muonValidation_step,
    process.MEtoEDMConverter_step,process.outpath)

