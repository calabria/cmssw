import FWCore.ParameterSet.Config as cms

processName = "MuonSuite"
process = cms.Process(processName)

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/user/calabria/Muminus_Pt100-gun/calabria_MuMinusPt100_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC12_2023Scenario_3Step/751194ca5d3aeb41ed7baca383591b5f/step3_1000_1_XSy.root',
       '/store/user/calabria/Muminus_Pt100-gun/calabria_MuMinusPt100_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC12_2023Scenario_3Step/751194ca5d3aeb41ed7baca383591b5f/step3_100_1_4Bm.root',
       '/store/user/calabria/Muminus_Pt100-gun/calabria_MuMinusPt100_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC12_2023Scenario_3Step/751194ca5d3aeb41ed7baca383591b5f/step3_101_1_akd.root',
       '/store/user/calabria/Muminus_Pt100-gun/calabria_MuMinusPt100_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC12_2023Scenario_3Step/751194ca5d3aeb41ed7baca383591b5f/step3_102_1_p0B.root',
       '/store/user/calabria/Muminus_Pt100-gun/calabria_MuMinusPt100_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC12_2023Scenario_3Step/751194ca5d3aeb41ed7baca383591b5f/step3_103_1_Vlo.root',
       '/store/user/calabria/Muminus_Pt100-gun/calabria_MuMinusPt100_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC12_2023Scenario_3Step/751194ca5d3aeb41ed7baca383591b5f/step3_104_1_ezk.root',
       '/store/user/calabria/Muminus_Pt100-gun/calabria_MuMinusPt100_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC12_2023Scenario_3Step/751194ca5d3aeb41ed7baca383591b5f/step3_105_1_2J8.root',
       '/store/user/calabria/Muminus_Pt100-gun/calabria_MuMinusPt100_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC12_2023Scenario_3Step/751194ca5d3aeb41ed7baca383591b5f/step3_106_1_jtP.root',
       '/store/user/calabria/Muminus_Pt100-gun/calabria_MuMinusPt100_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC12_2023Scenario_3Step/751194ca5d3aeb41ed7baca383591b5f/step3_107_1_dUf.root',
       '/store/user/calabria/Muminus_Pt100-gun/calabria_MuMinusPt100_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC12_2023Scenario_3Step/751194ca5d3aeb41ed7baca383591b5f/step3_108_1_CE4.root',
       '/store/user/calabria/Muminus_Pt100-gun/calabria_MuMinusPt100_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC12_2023Scenario_3Step/751194ca5d3aeb41ed7baca383591b5f/step3_109_1_xJ4.root',
       '/store/user/calabria/Muminus_Pt100-gun/calabria_MuMinusPt100_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC12_2023Scenario_3Step/751194ca5d3aeb41ed7baca383591b5f/step3_10_1_Zq4.root',
       '/store/user/calabria/Muminus_Pt100-gun/calabria_MuMinusPt100_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC12_2023Scenario_3Step/751194ca5d3aeb41ed7baca383591b5f/step3_110_1_UIl.root',
       '/store/user/calabria/Muminus_Pt100-gun/calabria_MuMinusPt100_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC12_2023Scenario_3Step/751194ca5d3aeb41ed7baca383591b5f/step3_111_1_nqX.root', ] );
secFiles.extend((

    ))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(200) )

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

process.schedule = cms.Schedule(
    #process.raw2digi_step,
    #process.validation_step,
    process.muonValidation_step,
    process.MEtoEDMConverter_step,process.outpath)

