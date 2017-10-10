#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Services/src/SiteLocalConfigService.h"
#include "FWCore/Services/src/JobReportService.h"
#include "FWCore/Services/interface/cuda_service.h"

using edm::service::JobReportService;
using edm::service::SiteLocalConfigService;
using edm::service::CudaService;

DEFINE_FWK_SERVICE(CudaService);

typedef edm::serviceregistry::ParameterSetMaker<edm::SiteLocalConfig,SiteLocalConfigService> SiteLocalConfigMaker;
DEFINE_FWK_SERVICE_MAKER(SiteLocalConfigService,SiteLocalConfigMaker);
typedef edm::serviceregistry::AllArgsMaker<edm::JobReport,JobReportService> JobReportMaker;
DEFINE_FWK_SERVICE_MAKER(JobReportService, JobReportMaker);
