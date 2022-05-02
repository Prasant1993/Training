#include "TROOT.h"
#include "TKey.h"
#include "TFile.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TObjArray.h"
#include "THStack.h"
#include "TLegend.h"
#include "TEfficiency.h"
#include "TGraphAsymmErrors.h"
#include "TF1.h"
#include "TMath.h"
#include "TCut.h"
#include "TColor.h"
#include "TPaletteAxis.h"
      
#include "TMVA/Reader.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <algorithm>
    
void plotMVARes(){

  string fileNames[15];

  
  fileNames[0]= "Ntuple/mvares_Hgg_phoId_EE_UL18_BDTG_TMVA_SATrain_SATest.root"; 
  fileNames[1]= "Ntuple/mvares_Hgg_phoId_EE_UL18_XGB_SATrain_SATest.root"; 
  fileNames[2]= "Ntuple/mvares_Hgg_phoId_EE_Autumn18_BDTG_TMVA_SATrain_SATest.root"; 


  //////////////////////////////////// Training wo showershape applied to Test wo showershape correction //////////////////////

  // extendTrain with SATest wHOE
  //fileNames[0]= "/afs/cern.ch/work/p/prrout/public/BDT_training_11062019/CMSSW_9_4_9/src/SA_train_wHOE_M95_13062019/Ntuple/mvares_Hgg_phoId_94X_EE_woCorrPhoIso_RunIIFall17_3_1_0_MCv2_extendTrain_M55_SATest_M95_woShowershape_nTree2k_BDT_wHOE.root";

  
  //fileNames[0]= "Ntuple_UL2017/mvares_Hgg_phoId_EE_LR_0p05_UL2017_SATrain_UL2017_SATest_BDTG_14052020.root";
  //fileNames[0]= "Ntuple_Summer16/mvares_Hgg_phoId_EE_Summer16_LMTrain_Summer16_LMTest_BDTG_12062020.root";
  //fileNames[1]= "Ntuple_Summer16/mvares_Hgg_phoId_EE_Fall17_LMTrain_Summer16_LMTest_BDTG_12062020.root";
  //fileNames[2]= "Ntuple_UL2017/mvares_Hgg_phoId_EE_Fall17_kuntal_SATrain_UL2017_SATest_BDT_14052020.root";
  //fileNames[1]= "Ntuple/mvares_Hgg_phoId_EE_RunIIAutumn18_xgboost_SATrain_woCorr_RnIIAutumn18_SATest_woCorr_M95_BDT.root";
  //fileNames[1]= "Ntuple/mvares_Hgg_phoId_EE_Kuntal_Fall17_SATrain_woCorr_RnIIAutumn18_SATest_woCorr_M95_BDT.root";
  // Prasant LMTrain with SATest wHOE
  //fileNames[2]= "/afs/cern.ch/work/p/prrout/public/BDT_training_11062019/CMSSW_9_4_9/src/SA_train_wHOE_M95_13062019/Ntuple/mvares_Hgg_phoId_94X_EE_woCorrPhoIso_RunIIFall17_3_1_0_MCv2_LMTrain_M55_SATest_M95_woShowershape_nTree2k_BDT_wHOE.root";

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // kuntal SATrain woHOE and SATest wHOE
  //fileNames[1] ="/afs/cern.ch/work/p/prrout/public/BDT_training_11062019/CMSSW_9_4_9/src/SA_train_wHOE_M95_13062019/Ntuple/mvares_Hgg_phoId_94X_EE_woCorrPhoIso_RunIIFall17_3_1_0_MCv2_SATrain_woHOE_SATest_wHOE_M95_woShowershape_nTree2k_BDT.root";

  // Kuntal SATrain woHOE and SATest wHOE
  //  fileNames[1]= "/eos/user/p/prrout/public/Flashgg/PhotonIDMVA/Extended_Space_05062019/Ntuple_wHOE_08062019_corrected/mvares_Hgg_phoId_94X_EE_woCorrPhoIso_RunIIFall17_3_1_0_MCv2_kuntalSATrain_PrasantSATest_woShowershape_nTree2k_BDTG.root";


  TCanvas * can = new TCanvas("can_mvares","can_mvares",600,600);
  string label_mvares = "mva output";

  TLegend *legend = new TLegend(0.35,0.55,0.8,0.85,"","brNDC");
  legend->SetHeader("PhoId, EE :");
  legend->SetBorderSize(0);
  legend->SetFillStyle(0);
  legend->SetTextFont(42);

  string labelLeg_s[15];
  labelLeg_s[0] = "Sig: UL2018_TMVA";
  labelLeg_s[1] = "Sig: UL2018_XGB";
  labelLeg_s[2] = "Sig: ReReco18_TMVA";


                                                                                                                                  
                                                                                                                                         
                   
  /*labelLeg_s[0] = "Sig: 94X LMTrain_LMTest_nTree2k_30pT18";
    labelLeg_s[1] = "Sig: 94X LMTrain_LMTest_nTree2k_18pT18";
    labelLeg_s[2] = "Sig: 94X SATrain_SATest_nTree2k";
    labelLeg_s[3] = "Sig: 94X SATrain_LMTest_nTree2k";                  
    labelLeg_s[4] = "Sig: 94X SATrain_SATest_LMPres_nTree2k";
    labelLeg_s[5] = "Sig: 94X SATrain_LMTest_SAPres_nTree2k";              
  */                                                      
  // labelLeg_s[1] = "Sig: 94X LMTest_LMTrain_LdSldpT18_nTree10k";  
  //labelLeg_s[2] = "Sig: 94X SATest_SATrain_nTree2k";
  //labelLeg_s[3] = "Sig: 94X LMTest_SATrain_nTree2k"; 

   
  
  /*
  labelLeg_s[0] = "Sig: 94X LMTest_LMTrain_nTree2k";
  labelLeg_s[1] = "Sig: 94X LMTest_LMTrain_nTree4k";
  labelLeg_s[2] = "Sig: 94X LMTest_LMTrain_nTree6k";
  labelLeg_s[3] = "Sig: 94X LMTest_LMTrain_nTree8k";
  labelLeg_s[4] = "Sig: 94X LMTest_LMTrain_nTree2k_Alt_NEvt";
  labelLeg_s[5] = "Sig: 94X LMTest_LMTrain_nTree4k_Alt_NEvt";
  labelLeg_s[6] = "Sig: 94X LMTest_LMTrain_nTree8k_MnNdsz5%_Dph3";
  labelLeg_s[7] = "Sig: 94X LMTest_LMTrain_nTree10k_MnNdsz5%_Dph3";
  labelLeg_s[8] = "Sig: 94X LMTest_SATrain_nTree2k";
  labelLeg_s[9] = "Sig: 94X SATest_SATrain_nTree2k";
  */

  //labelLeg_s[0] = "Sig: 94X LMTrain_LMTest_Mgg>55";
  //labelLeg_s[1] = "Sig: 94X SATrain_SATest_Mgg>95";
  //labelLeg_s[2] = "Sig: 94X SATrain_LMTest_Mgg>55";
  //labelLeg_s[3] = "Sig: 94X SATest_LMTrain_LMWeight";
  //labelLeg_s[4] = "Sig: 94X LMTest_SATrain_LMWeight";
  //labelLeg_s[3] = "Sig: 94X LMTest_LMTrain_LMWeight_nTree4k";
  //labelLeg_s[4] = "Sig: 94X LMTest_LMTrain_LMWeight_nTree6k"; 
  //labelLeg_s[3] = "Sig: 94X LMTrain_LMTest_Mgg>60";
  //labelLeg_s[4] = "Sig: 94X LMTrain_LMTest_Mgg>65";
  //labelLeg_s[5] = "Sig: 94X LMTrain_LMTest_Mgg>80";
  //labelLeg_s[6] = "Sig: 94X LMTrain_LMTest_Mgg>95";
  //labelLeg_s[7] = "Sig: 94X SATrain_LMTest_Mgg>80";
  //labelLeg_s[8] = "Sig: 94X SATrain_LMTest_Mgg>95";


  string labelLeg_b[15];
  labelLeg_b[0] = "Bkg: UL2018_TMVA";
  labelLeg_b[1] = "Bkg: UL2018_XGB";  
  labelLeg_b[2] = "Bkg: ReReco18_TMVA";

  //labelLeg_b[0] = "Bkg: 94X LMTrain_LMTest_nTree2k_30pt18";
  /*labelLeg_b[0] = "Bkg: 94X LMTrain_LMTest_nTree2k_30pT18";
  labelLeg_b[1] = "Bkg: 94X LMTrain_LMTest_nTree2k_30pT18";
  labelLeg_b[2] = "Bkg: 94X SATrain_SATest_nTree2k";
  labelLeg_b[3] = "Bkg: 94X SATrain_LMTest_nTree2k";
  labelLeg_b[4] = "Bkg: 94X SATrain_SATest_LMPres_nTree2k";
  labelLeg_b[5] = "Bkg: 94X SATrain_LMTest_SAPres_nTree2k";
  */

  //labelLeg_b[1] = "Bkg: 94X LMTest_LMTrain_LdSldpT18_nTree10k";
  //labelLeg_b[2] = "Bkg: 94X SATest_SATrain_nTree2k";
  //labelLeg_b[3] = "Bkg: 94X LMTest_SATrain_nTree2k";

  /*
  labelLeg_b[0] = "Bkg: 94X LMTest_LMTrain_nTree2k";
  labelLeg_b[1] = "Bkg: 94X LMTest_LMTrain_nTree4k";
  labelLeg_b[2] = "Bkg: 94X LMTest_LMTrain_nTree6k";
  labelLeg_b[3] = "Bkg: 94X LMTest_LMTrain_nTree8k";
  labelLeg_b[4] = "Bkg: 94X LMTest_LMTrain_nTree2k_Alt_NEvt";
  labelLeg_b[5] = "Bkg: 94X LMTest_LMTrain_nTree4k_Alt_NEvt";
  labelLeg_b[6] = "Bkg: 94X LMTest_LMTrain_nTree8k_MnNdsz5%_Dph3";
  labelLeg_b[7] = "Bkg: 94X LMTest_LMTrain_nTree10k_MnNdsz5%_Dph3";
  labelLeg_b[8] = "Bkg: 94X LMTest_SATrain_nTree2k";
  labelLeg_b[9] = "Bkg: 94X SATest_SATrain_nTree2k";
  */
  //labelLeg_b[0] = "Bkg: 94X LMTrain_LMTest_Mgg>55";
  //labelLeg_b[1] = "Bkg: 94X SATrain_SATest_Mgg>95";
  //labelLeg_b[2] = "Bkg: 94X SATrain_LMTest_Mgg>55";
  //labelLeg_b[3] = "Bkg: 94X SATest_LMTrain_LMWeight";
  //labelLeg_b[4] = "Bkg: 94X LMTest_SATrain_LMWeight";
  //labelLeg_b[3] = "Bkg: 94X LMTest_LMTrain_LMWeight_nTree4k";
  //labelLeg_b[4] = "Bkg: 94X LMTest_LMTrain_LMWeight_nTree6k";
  //labelLeg_b[3] = "Bkg: 94X LMTrain_LMTest_Mgg>60";
  //labelLeg_b[4] = "Bkg: 94X LMTrain_LMTest_Mgg>65";
  //labelLeg_b[5] = "Bkg: 94X LMTrain_LMTest_Mgg>80";
  //labelLeg_b[6] = "Bkg: 94X LMTrain_LMTest_Mgg>95";
  //labelLeg_b[7] = "Bkg: 94X SATrain_LMTest_Mgg>80";
  //labelLeg_b[8] = "Bkg: 94X SATrain_LMTest_Mgg>95";



  TCanvas * can_RoC = new TCanvas ("can_RoC","can_RoC",600,600);

  TLegend *legend_RoC = new TLegend(0.2,0.5,0.6,0.90,"","brNDC");
  legend_RoC->SetHeader("PhoId, EE :");
  legend_RoC->SetBorderSize(0);
  legend_RoC->SetFillStyle(0);
  legend_RoC->SetTextFont(42);

  string labelLeg_RoC[15];  
  labelLeg_RoC[0] = "UL2018_TMVA"; 
  labelLeg_RoC[1] = "UL2018_XGB";
  labelLeg_RoC[2] = "ReReco18_TMVA";
                                                                                                                                          
                 
  //labelLeg_RoC[0] = "94X LMTrain_LMTest_nTree2k_30pt18";

  /*labelLeg_RoC[0] = "94X LMTrain_LMTest_nTree2k_30pT18";
  labelLeg_RoC[1] = "94X LMTrain_LMTest_nTree2k_18pT18";
  labelLeg_RoC[2] = "94X SATrain_SATest_nTree2k";
  labelLeg_RoC[3] = "94X SATrain_LMTest_nTree2k";
  labelLeg_RoC[4] = "94X SATrain_SATest_LMPres_nTree2k";
  labelLeg_RoC[5] = "94X SATrain_LMTest_SAPres_nTree2k";
  */

//labelLeg_RoC[1] = "94X LMTest_LMTrain_LdSldpT18_nTree10k";
  //labelLeg_RoC[2] = "94X SATest_SATrain_nTree2k";
  //labelLeg_RoC[3] = "94X LMTest_SATrain_nTree2k";

  /*
  labelLeg_RoC[0] = "94X LMTest_LMTrain_nTree2k";
  labelLeg_RoC[1] = "94X LMTest_LMTrain_nTree4k";
  labelLeg_RoC[2] = "94X LMTest_LMTrain_nTree6k";
  labelLeg_RoC[3] = "94X LMTest_LMTrain_nTree8k";
  labelLeg_RoC[4] = "94X LMTest_LMTrain_nTree2k_Alt_NEvt";
  labelLeg_RoC[5] = "94X LMTest_LMTrain_nTree4k_Alt_NEvt";
  labelLeg_RoC[6] = "94X LMTest_LMTrain_nTree8k_MnNdesz5%_Dph3";
  labelLeg_RoC[7] = "94X LMTest_LMTrain_nTree10k_MnNdsz5%_Dph3";
  labelLeg_RoC[8] = "94X LMTest_SATrain_nTree2k";
  labelLeg_RoC[9] = "94X SATest_SATrain_nTree2k";

  */

  //labelLeg_RoC[0] = "94X LMTrain_LMTest_Mgg>55";
  //labelLeg_RoC[1] = "94X SATrain_SATest_Mgg>95";
  //labelLeg_RoC[2] = "94X SATrain_LMTest_Mgg>55";
  //labelLeg_RoC[3] = "94X SATest_LMTrain_LMWeight";
  //labelLeg_RoC[4] = "94X LMTest_SATrain_LMWeight";
  //labelLeg_RoC[3] = "94X LMTest_LMTrain_LMWeight_nTree4k";
  //labelLeg_RoC[4] = "94X LMTest_LMTrain_LMWeight_nTree6k";
  //labelLeg_RoC[3] = "94X LMTrain_LMTest_Mgg>60";
  //labelLeg_RoC[4] = "94X LMTrain_LMTest_Mgg>65";
  //labelLeg_RoC[5] = "94X LMTrain_LMTest_Mgg>80";
  //labelLeg_RoC[6] = "94X LMTrain_LMTest_Mgg>95";
  //labelLeg_RoC[7] = "94X SATrain_LMTest_Mgg>80";
  //labelLeg_RoC[8] = "94X SATrain_LMTest_Mgg>95";

  for(int i = 0; i < 3; i++){

    cout << "file # " << i << endl;

    TFile *mvaResFile = new TFile(fileNames[i].c_str());

    TTree *t_output_s = (TTree*)mvaResFile->Get("promptPhotons");
    TTree *t_output_b = (TTree*)mvaResFile->Get("fakePhotons");

    TH1F * histo_s = new TH1F ("histo_s","histo_s",2000,-1,1);
    TH1F * histo_b = new TH1F ("histo_b","histo_b",2000,-1,1);

    TString tmp_s = "";
    TString tmp_b  = "";

    tmp_s = "mvares";
    tmp_s+=">>histo_s";
    
    tmp_b = "mvares";
    tmp_b+=">>histo_b";

    
    if(i == 0){
      t_output_s->Draw(tmp_s,"(abs(scEta)>1.5)*weight","goff");
      t_output_b->Draw(tmp_b,"(abs(scEta)>1.5)*weight","goff");
    }
    else{
      t_output_s->Draw(tmp_s,"(abs(scEta)>1.5)*weight","goff");
      t_output_b->Draw(tmp_b,"(abs(scEta)>1.5)*weight","goff");
    }

   
    float Nsig[20000], Nbkg[20000];
    float sigEff[20000], bkgEff[20000], bkgrejEff[20000];
    float cutsVal[20000];
    float mvaResCutVal = -1.0001;

    int nCuts = 20000;

    int mvaSMaxBin = histo_s->GetXaxis()->FindBin(1);
    int mvaBMaxBin = histo_b->GetXaxis()->FindBin(1);

    for(int k = 0; k < nCuts; k++){

      mvaResCutVal+= 0.0001;
      cutsVal[k] = mvaResCutVal;

      int mvaBin = histo_s->GetXaxis()->FindBin(mvaResCutVal);
      Nsig[k] = histo_s->Integral(mvaBin,mvaSMaxBin);

      int mvaBin_b = histo_b->GetXaxis()->FindBin(mvaResCutVal);
      Nbkg[k] = histo_b->Integral(mvaBin_b,mvaBMaxBin);
      sigEff[k] = Nsig[k]/Nsig[0];
      bkgEff[k] = Nbkg[k]/Nbkg[0];
      bkgrejEff[k] = 1 - bkgEff[k];

      //if(sigEff[k] >= 0.99)  
      //cout << " sigEff[k] = " << sigEff[k] <<  " bkgEff[k] = " << bkgEff[k] << " with a cut at " << mvaResCutVal << endl;
      //if(sigEff[k] > 0.89 && sigEff[k] < 0.91) cout << " sigEff[k] = " << sigEff[k] <<  " bkgEff[k] = " << bkgEff[k] << " with a cut at " << mvaResCutVal << endl;
      /*std::fstream outfile;
      if(i==0){
        outfile.open("sigEff_bkgEff_endcap_UL18_TMVA_SATrain_UL18_SATest_20000_nbins2000.txt",std::fstream::out | std::fstream::app);
        outfile << bkgEff[k] <<'\t'<< sigEff[k] <<'\t'<< bkgrejEff[k] <<'\t'<<  mvaResCutVal << endl;
	} 

      if(i==1){
        outfile.open("sigEff_bkgEff_endcap_UL18_XGB_SATrain_UL18_SATest_20000_nbins2000.txt",std::fstream::out | std::fstream::app);
        outfile << bkgEff[k] <<'\t'<< sigEff[k] <<'\t'<< bkgrejEff[k] <<'\t'<<  mvaResCutVal << endl;
	}
      
      if(i==2){
        outfile.open("sigEff_bkgEff_endcap_ReReco18_TMVA_SATrain_UL18_SATest_20000_nbins2000.txt",std::fstream::out | std::fstream::app);
        outfile << bkgEff[k] <<'\t'<< sigEff[k] <<'\t'<< bkgrejEff[k] <<'\t'<<  mvaResCutVal << endl;
	} 
      */


	}

    TGraph * sigEff_vs_cut = new TGraph (nCuts, cutsVal, sigEff);
    TGraph * bkgEff_vs_cut = new TGraph (nCuts, cutsVal, bkgEff);
    TGraph * sigEff_vs_bkgEff = new TGraph (nCuts, sigEff, bkgEff);

    //draw mvares
    can->cd();
    can->SetLogy();

    histo_s->SetTitle("");
    histo_s->SetStats(0);
    histo_s->GetXaxis()->SetTitle(label_mvares.c_str());
    histo_s->GetYaxis()->SetTitle("Events/0.02");
    histo_s->SetMaximum(histo_s->GetBinContent(histo_s->GetMaximumBin())*1000);
       
    //histo_s->SetMarkerColor(i+2);
    //histo_s->SetLineColor(i+2);
    histo_s->SetLineWidth(2);

    histo_b->SetLineStyle(2);
    //histo_b->SetMarkerColor(i+2);
    //histo_b->SetLineColor(i+2);
    histo_b->SetLineWidth(2);

    if(i == 2) {
      histo_s->SetLineColor(kGreen+2);
      histo_s->SetMarkerColor(kGreen+2);
      histo_b->SetLineColor(kGreen+2);
      histo_b->SetMarkerColor(kGreen+2);
    }
    if(i == 1){
      histo_s->SetLineColor(kBlue);
      histo_s->SetMarkerColor(kBlue);
      histo_b->SetLineColor(kBlue);
      histo_b->SetMarkerColor(kBlue);
    }
    
    if(i == 0){
      histo_s->SetLineColor(kRed);
      histo_s->SetMarkerColor(kRed);
      histo_b->SetLineColor(kRed);
      histo_b->SetMarkerColor(kRed);
    }
    
    if(i == 3){
      histo_s->SetLineColor(5);
      histo_s->SetMarkerColor(5);
      histo_b->SetLineColor(5);
      histo_b->SetMarkerColor(5);
    }
    if(i == 4){
      histo_s->SetLineColor(6);
      histo_s->SetMarkerColor(6);
      histo_b->SetLineColor(6);
      histo_b->SetMarkerColor(6);
    }
    if(i == 5){
      histo_s->SetLineColor(7);
      histo_s->SetMarkerColor(7);
      histo_b->SetLineColor(7);
      histo_b->SetMarkerColor(7);
    }
    if(i == 6){
      histo_s->SetLineColor(8);
      histo_s->SetMarkerColor(8);
      histo_b->SetLineColor(8);
      histo_b->SetMarkerColor(8);
    }
    if(i == 7){
      histo_s->SetLineColor(9);
      histo_s->SetMarkerColor(9);
      histo_b->SetLineColor(9);
      histo_b->SetMarkerColor(9);
    }
    if(i == 8){
      histo_s->SetLineColor(46);
      histo_s->SetMarkerColor(46);
      histo_b->SetLineColor(46);
      histo_b->SetMarkerColor(46);
    }
    if(i == 9){
      histo_s->SetLineColor(kMagenta+4);
      histo_s->SetMarkerColor(kMagenta+4);
      histo_b->SetLineColor(kMagenta+4);
      histo_b->SetMarkerColor(kMagenta+4);
    }
    
    /*     else{
      histo_s->SetLineColor(i+2);
      histo_s->SetMarkerColor(i+2);
      histo_b->SetLineColor(i+2);
      histo_b->SetMarkerColor(i+2);
    }
    */ 

    legend->AddEntry(histo_s,labelLeg_s[i].c_str(),"lem");
    legend->AddEntry(histo_b,labelLeg_b[i].c_str(),"lem");

        
    if(i == 0){
      histo_s->Draw("HIST");
    }
    else histo_s->Draw("HISTSAME");
    legend->Draw("same");
    histo_b->Draw("same");

    TLatex *txt = new TLatex(0.2, 0.9, "");
    // txt->SetTextSize(0.05);                                                                                                                

 
    txt->DrawLatexNDC(0.1, 0.91, "CMS #bf{#it{#scale[0.8]{Simulation Preliminary}}}");
    txt->DrawLatexNDC(0.76, 0.91, "#bf{13 TeV}");
    txt->Draw("same");


    can->Update();
    can->Modified();

    //draw RoC curves 

    can_RoC->cd();

    if(i == 0){
      sigEff_vs_bkgEff->SetTitle("");
      sigEff_vs_bkgEff->GetYaxis()->SetTitleOffset(1.5);
      sigEff_vs_bkgEff->GetYaxis()->SetTitleSize(0.03);
      sigEff_vs_bkgEff->GetYaxis()->SetLabelSize(0.02);
      sigEff_vs_bkgEff->GetXaxis()->SetTitleOffset(1.5);
      sigEff_vs_bkgEff->GetXaxis()->SetTitleSize(0.03);
      sigEff_vs_bkgEff->GetXaxis()->SetLabelSize(0.02);

      
      //sigEff_vs_bkgEff->GetXaxis()->SetRangeUser(0.8,1.);
      //sigEff_vs_bkgEff->GetYaxis()->SetRangeUser(0.05,0.3);
      
      sigEff_vs_bkgEff->GetXaxis()->SetRangeUser(0.8,1.0);
      sigEff_vs_bkgEff->GetYaxis()->SetRangeUser(0.0,1.0);

      sigEff_vs_bkgEff->GetXaxis()->SetTitle("signal eff");
      sigEff_vs_bkgEff->GetYaxis()->SetTitle("bkg eff");

    }


    sigEff_vs_bkgEff->SetLineWidth(2);

    if(i == 1) {
      sigEff_vs_bkgEff->SetLineColor(kBlue);
      sigEff_vs_bkgEff->SetMarkerColor(kBlue);
    }
    if(i == 2){
      sigEff_vs_bkgEff->SetLineColor(kGreen+2);
      sigEff_vs_bkgEff->SetMarkerColor(kGreen+2);
    }
    if(i == 0){
      sigEff_vs_bkgEff->SetLineColor(kRed);
      sigEff_vs_bkgEff->SetMarkerColor(kRed);
    }
    
    if(i == 3){
      sigEff_vs_bkgEff->SetLineColor(5);
      sigEff_vs_bkgEff->SetMarkerColor(5);
    }
    if(i == 4){
      sigEff_vs_bkgEff->SetLineColor(6);
      sigEff_vs_bkgEff->SetMarkerColor(6);
    }
    if(i == 5){
      sigEff_vs_bkgEff->SetLineColor(7);
      sigEff_vs_bkgEff->SetMarkerColor(7);
    }
    if(i == 6){
      sigEff_vs_bkgEff->SetLineColor(8);
      sigEff_vs_bkgEff->SetMarkerColor(8);
    }
    if(i == 7){
      sigEff_vs_bkgEff->SetLineColor(9);
      sigEff_vs_bkgEff->SetMarkerColor(9);
    }
    if(i == 8){
      sigEff_vs_bkgEff->SetLineColor(46);
      sigEff_vs_bkgEff->SetMarkerColor(46);
    }
    if(i == 9){
      sigEff_vs_bkgEff->SetLineColor(kMagenta+4);
      sigEff_vs_bkgEff->SetMarkerColor(kMagenta+4);
    }
    
    
    /*else{
      sigEff_vs_bkgEff->SetLineColor(i+2);
      sigEff_vs_bkgEff->SetMarkerColor(i+2);
    }
    */ 

    can_RoC->SetGrid();

    if(i == 0)  sigEff_vs_bkgEff->Draw("AC");
    else sigEff_vs_bkgEff->Draw("sameC");

    legend_RoC->AddEntry(sigEff_vs_bkgEff,labelLeg_RoC[i].c_str(),"pl");
    legend_RoC->Draw("same");

    TLatex *txt1 = new TLatex(0.2, 0.9, "");
    // txt->SetTextSize(0.05);                                                                                                                

    txt1->DrawLatexNDC(0.1, 0.91, "CMS #bf{#it{#scale[0.8]{Simulation Preliminary}}}");
    txt->DrawLatexNDC(0.76, 0.91, "#bf{13 TeV}");
    txt1->Draw("same");

    can_RoC->Update();
    can_RoC->Modified();


  }

  string mvaRes = "";
  mvaRes = "mvares_EE_phoId_Hgg_UL18_SATest_20000_nbins_2000_V1";

  can->SaveAs((mvaRes+".pdf").c_str()); 
  can->SaveAs((mvaRes+".png").c_str()); 
  can->SaveAs((mvaRes+".root").c_str()); 

  string canName_RoC = "";

  canName_RoC = "RoC_EE_phoId_Hgg_UL18_SATest_20000_nbins_2000_0p8range_V2";

  can_RoC->SaveAs((canName_RoC+".pdf").c_str()); 
  can_RoC->SaveAs((canName_RoC+".png").c_str()); 
  can_RoC->SaveAs((canName_RoC+".root").c_str()); 
}


