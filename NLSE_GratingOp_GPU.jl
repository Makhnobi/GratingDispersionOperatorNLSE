using CUDA #for gpu you need CUDA.jl
#the solver
function gpu_NLSE_BG(ω,input_field,β0,β1,β2,β3,β4,ω_REF,distance,nt,step_num,c1,Λ_µm;isgrating=true,Aeff=1e-12)
    #What is the purpose of this function ?
    #Given an input pulse (spectral), you will obtain the output spectrum (and temporal shape) after propagation in a waveguide/grating defined by its parameters (effective area, dispersion  β(ω), length...).
    #The current code include only Kerr effect, but it is possible to inclue Raman response or other supplementary nonlinear effects with a bit of work.
  
    #OUTPUT : N_c (nonlinear coefficient \gamma)
              #dispersion is the function that calculates the dispersion (think wave vector influence)
              #uu_3d is the temporal output of the guide
              #spect_3d is the spectral output of the guide

    #INPUT PARAMETERS
    #ω is the angular frequency vector (ω=2πc/λ in rad.Hz) and is supposed well defined (ex:ωg = 1e9*2*pi*c/1300; ωd = 1e9*2*pi*c/1800;ω=LinRange(ωd,ωg,nb_pts);  )
    #input_field is your input pulse (spectral domain) and is supposed well defined
    ##there is a function to create the pulse you want later, do not worry.

    #β0,β1,β2,β3,β4 are the Taylor coefficients that forms the dispersion β(ω)
    #β2 being the Group-velocity dispersion (GVD)
    #ω_REF is the reference frequency (for the Taylor decomposition)
    ##and for now    MUST   be equal to the Bragg frequency (easier formulation)
    ω_Bragg=ω_REF;
    #distance (in m) is the propagation length in the waveguide
    #nt is your number of temporal points in the temporal wndow (also number of spectral points in the spectral one)
    #step_num is the number of step in the propagation direction

    #grating specific parameters:
    #c1 is the first coefficent of the Fourier decomposition of the index modulation.
    ##can be approximated by δn/2
    #Λ_µm is the grating period in µm
    #isgrating serves as an indicator for the algorithm
    ## to know if we want the grating output or the equivalent waveguide output (no index modulation)

    #Aeff (constant in m^2) the effective area 10^-12 m^2  by default
    c=3e8; #speed of light
    n2=2.4e-19; #nonlinear index typical in SiN 2.4e-19;
    #functino to retrieve wavevector from Taylor coefficients
    fct_β=x->β0.+β1.*x.+β2.*x.^2 ./2 .+β3.*x.^3 ./6 .+β4.*x.^4 ./24;
    δz = distance/step_num; # step size in z
    
    #the grating dispersion operator
    df=(q,d,z)->q.*(d.*cos.(q*z).-im*q.*sin.(q*z))./(q.*cos.(q*z).-im*d.*sin.(q*z));#q
    
    #making everything work on GPU:    
    ω_gpu=CUDA.CuArray{Float32}(undef,nt);
    @. ω_gpu=ω;
    ω_β=(ω_gpu.-ω_REF); #taylor
    #λ_bragg=2*pi*c/ω_Bragg;
    β=fct_β(ω_β);
    β_gpu=CUDA.CuArray{Float32}(undef,nt);
    @. β_gpu=β;
    δ_nn_eval=CUDA.CuArray{Float32}(undef,nt);
    κ_nn_eval=CUDA.CuArray{Float32}(undef,nt);
    q_nn_eval=CUDA.CuArray{ComplexF32}(undef,nt);
    (v,index_Bragg)=findmin(abs.(ω.-ω_Bragg));
    ipf=ComplexF32.(input_field);
    if (isgrating) #if grating
        β0g=1e6*pi/Λ_µm;# =n_eff_Bragg*ω_Bragg/c par def =β0 si ref et bragg fqcy equal  
        @. δ_nn_eval=β_gpu-β0g;
        @. κ_nn_eval=ω*c1/c;
        @. q_nn_eval=sqrt(ComplexF32(δ_nn_eval^2 -κ_nn_eval^2));      
        @. q_nn_eval=q_nn_eval;
        dispersion = z -> exp.(im.*δz.*(df(q_nn_eval,δ_nn_eval,z).-β1*ω_β));

        #wrong dispersion operator from westbrook 2006 below:
        #f=(q,d,z)-> -im*log.(im*q./(im*q.*cos.(q*z).+d.*sin.(q*z)));
        #dispersion = z -> exp.(im.*propagation_length_m.*(f(q_nn_eval,δ_nn_eval,propagation_length_m).-β1*ω_β));
        sss=Int.(abs.(δ_nn_eval)./δ_nn_eval); #sign function
    else
        arg_dispersion=β.-β0.-β1.*ω_β; #.-β1.*ω_β no β1 <=> moving frame of central wavelength ?
        f=(q,d,z)-> exp.(im.*δz.*(arg_dispersion));
        #println("grating gap outside considered range => neglected")       
        dispersion = z -> f(0,0,0);#constante donc osef
    end
    
    vvv=CUDA.CuArray{ComplexF32}(undef,nt);
    copyto!(vvv,ipf);
    uu=CUDA.CuArray{ComplexF32}(undef,nt);
    uu .= CUDA.CUFFT.ifft(vvv); #uu = fftshift(ifft(vvv)); #def pulse en freq
    uu .= CUDA.CUFFT.ifftshift(uu);
    N_c=CUDA.CuArray{ComplexF32}(undef,nt);
    @. N_c=n2*ω/(c*Aeff); # = γ
    #-----Input 3D data 
    uu_3d=CUDA.CuArray{ComplexF32}(undef,nt);
    spect_3d=CUDA.CuArray{ComplexF32}(undef,nt);
    temp=CUDA.CuArray{ComplexF32}(undef,nt);
    hhz=CUDA.CuArray{ComplexF32}(undef,nt);
    @. hhz = ComplexF32(im*δz*N_c);
    gpu_disper_vect=CUDA.CuArray{ComplexF32}(undef,nt);
    f_temp=CUDA.CuArray{ComplexF32}(undef,nt);
    
    #*********[ Beginning of MAIN Loop]***********
    # scheme: 1/2N -> D -> 1/2N; first half step nonlinear
    @. temp = uu*exp(abs2(uu) * hhz/2.0); #hhz/2 :)
    FFT_op=CUDA.CUFFT.plan_fft(uu);
    iFFT_op=CUDA.CUFFT.plan_ifft(uu);
    
    for n=1:step_num-1
      gpu_disper_vect.=dispersion(n*δz);
      f_temp .=FFT_op*temp.*gpu_disper_vect;
      uu.=iFFT_op*f_temp
      @. temp = (uu*exp((abs2.(uu))*hhz));
    end  
    gpu_disper_vect.=dispersion(step_num*δz);
    f_temp .=FFT_op*temp.*gpu_disper_vect;
    uu.=iFFT_op*f_temp
    @. temp = (uu*exp(((abs2(uu)))*hhz/2.0));
    spect_3d.=FFT_op*temp;
    uu_3d.=reverse(temp);#reverse(temp.*conj.(temp)); # temporal power     because Fourier convention is different from Agrawal (t-> -t)
    return (N_c,dispersion,uu_3d,spect_3d)
end

#how to build an inut spectrum ? See below ! (Limited to 4 input frequencies in this version, but can be generalized if needed).
#I included some function to help.

function gauss(x)
    return (exp.(-x.^(2)))
end

function gauss8(x)
    return exp.(-x.^(8))
end 
#to define input pulse

function white(x,facteur=1e-16)
    #for float only randn for white guassian
    #h=6.62607015e-34
    return (1 .+ facteur.*rand())./2
end

function momentum(x)
    return exp.(-im*x)
end

#the key function :
function input_pulse(ω,λ1,λ2,λ3,λ4,t_pulse=[50000;5000;5000;50000],Ampli=[20;20;1;20];f1=gauss8,f2=gauss,f3=gauss,f4=gauss8,noise=0)
    #build the input pulse sat the desired wavelengths with desired amplitudes so that temporal pulse (modulus squared) in in W, and spectrum (m.s.) in W/Hz
    # the 4 pulses are considered having a negligible group velocity mismatch (GVM) so that we can considered them as a single megapulse with a lot of frequencies.

    #ω is the (angular) frequency extent of the spectral window. It is an ARRAY that goes from minimal frequency to maximal frequency.
    #Ampli in sqrt(Watts) is an ARRAY containing the 4 amplitude in W of the 4 pulse (can be 0)
    #t_pulse in fs is an ARRAY containing the 4 temporal width (FWHM) in fs of the 4 pulse (can be 0)
    #f1,f2... are the FUNCTION that give the shapes of the pulses.
    #noise indicates if there is supplementary noise or not. It is a FLOAT and there is noise if it is not 0. Changes in the nature/amplitude of the noise in NOT implemented.
    c=3e8;
    A1=Ampli[1];A2=Ampli[2];A3=Ampli[3];A4=Ampli[4];
    t1_fs=t_pulse[1];t2_fs=t_pulse[2];t3_fs=t_pulse[3];t4_fs=t_pulse[4];
    if ω[1]<1e14
        println("warning: frequencies seem too low")
    end
    nnn=length(ω);
    dt=2*pi/((ω[2]-ω[1])*nnn);
    if λ1<1 #human entered all wavelengths in the same sub-unit
        ω1=2*pi*c/λ1;     
        ω2=2*pi*c/λ2;
        ω3=2*pi*c/λ3;
        ω4=2*pi*c/λ4;
    else
        ω1=2e9*pi*c/λ1;     
        ω2=2e9*pi*c/λ2;
        ω3=2e9*pi*c/λ3;
        ω4=2e9*pi*c/λ4;
    end
    t1=t1_fs.*1e-15;
    t2=t2_fs.*1e-15;
    t3=t3_fs.*1e-15;
    t4=t4_fs.*1e-15;

    omega1=((ω.-ω1)./(2*pi).*t1)*pi./(2*sqrt(log(2))); #argt dans la fct f1 #en pulsation ((ω.-ω1).*t1)./(4*log(2));
    omega2=((ω.-ω2)./(2*pi).*t2)*pi./(2*sqrt(log(2))); #argt ds la fct f2
    omega3=((ω.-ω3)./(2*pi).*t3)*pi./(2*sqrt(log(2))); #etc
    omega4=((ω.-ω4)./(2*pi).*t4)*pi./(2*sqrt(log(2)));
    p1=map(f1,omega1)*A1*t1*sqrt(pi/log(2))/(2*dt);#+ un peu de bruit
    p2=map(f2,omega2)*A2*t2*sqrt(pi/log(2))/(2*dt);
    p3=map(f3,omega3)*A3*t3*sqrt(pi/log(2))/(2*dt);
    p4=map(f4,omega4)*A4*t4*sqrt(pi/log(2))/(2*dt);
    p=p1.+p2.+p3.+p4.+(noise!=0).*(sqrt.((rand(nnn)*0.75.+0.25).*6.62607015e-34.*ω./(2*pi)).*exp.(im.*2*pi.*rand(nnn)));
    pt=ifftshift(ifft(p))
    return p,pt,A1,A2,A3,A4
    #time centered + spectrum centered
    #https://dsp.stackexchange.com/questions/28684/gaussian-wavelet-generation-of-a-given-frequency/28698#28698
end


#example of use: in a separate file just copy/paste what is below and uncomment.
#include("NLSE_GratingOp_GPU.jl") #or using NLSE_GratingOp_GPU
##data are from physiscal simulation I did on my side and holds a relevant physical value. 

#=


##parameters
λ_Bragg=1548.48172;δn1=2.245e-4
ω_Bragg=omega(λ_Bragg);
(β0_ssf,β1_ssf,β2_ssf,β3_ssf,β4_ssf)=(7.221901676788411e6, 7.103586631031164e-9, -3.217220630326691e-25, 4.630948732894057e-40, -1.7966102300411237e-55);
Λ_µm=0.435;nt=2^20;nz=1024;wg_length=1.0*1e-2;L=wg_length;
λ1=1546.35;λ2=1544.23023;λ4=1502;λ3=1524.945;
λ_start=1300; λ_stop=1800;
ω=LinRange(2e9*pi*c/λ_stop,2e9*pi*c/λ_start,nt);
U1=1*sqrt(81/2);U2=1*sqrt(1e-1);U3=0*sqrt(1e-1);U4=0*sqrt(81/2);#sqrt temporal amplitude of each
Amplitudes=[U1;U2;U3;U4];
t_pulse=[200000;100000;100000;200000];#in fs

## write the pulse
(i_f,i_t,A1t,A2t,A3t,A4t)=input_pulse(ω,λ1,λ2,λ3,λ4,t_pulse,Amplitudes;f1=gauss,f4=gauss,f3=gauss,noise=1);

## get the output

## solve NLSE in grating
(γ,Dispersion,uu_3dgpu,spect_3dgpu)=gpu_NLSE_BG(ω,i_f,β0_ssf,β1_ssf,β2_ssf,β3_ssf,β4_ssf,ω_REF,L,nt,nz,δn1,Λ_µm;isgrating=true,Aeff=1e-12);
spect_3dg=Array(spect_3dgpu);uu_3dg=Array(uu_3dgpu);
## in equivalent waveguide
(γ,Dispersion,uu_3wgpu,spect_3wgpu)=gpu_NLSE_BG(ω,i_f,β0_ssf,β1_ssf,β2_ssf,β3_ssf,β4_ssf,ω_REF,L,nt,nz,δn1,Λ_µm;isgrating=false,Aeff=1e-12);
spect_wg=Array(spect_3wgpu);uu_wg=Array(uu_3dgpu);




##BONUS if you want a nice plot (require CairoMakie.jl)


using CairoMakie
_,ind_signal=findmin(abs.(ω.-omega(λ2)));
_,ind_idler=findmin(abs.(ω.-omega(λ_Bragg)));
gauche=ind_idler-2000;droite=ind_idler+4500;g_signal=ind_signal-4000;d_signal=ind_signal+2000; 
fig=Figure(; resolution=(1000, 350));
axes1=Axis(fig[1,1],limits=((nothing,nothing),(-50,-0)),xlabel="frequency(THz)", ylabel = rich("η=P",subscript("out"),"(λ)/P",subscript("in"),rich("(λ",subscript("signal"),")"), " (dB)"),xlabelsize=22,ylabelsize=22,xticklabelsize=22,yticklabelsize=22)
CairoMakie.lines!(axes1,ω[gauche:droite]./(2e12*pi),10 .*log10.(abs2.(spect_wg[gauche:droite]./i_f[ind_signal])),linewidth=3,color=RGBf(0.0,0.9,1.0),label="output \nwaveguide")
CairoMakie.lines!(axes1,ω[gauche:droite]./(2e12*pi),10 .*log10.(abs2.(spect_3dg[gauche:droite]./i_f[ind_signal])),linewidth=3,color=RGBf(0.0,0.2,1.0),label="output \ngrating")
CairoMakie.scatterlines!(axes1,ω[gauche:droite]./(2e12*pi),10 .*log10.(abs2.(i_f[gauche:droite]./i_f[ind_signal])),color=RGBf(1.0,0.5,0.2),linewidth=2,markersize=2,label="input");#real c'(z) donne vect onde
CairoMakie.vlines!(axes1,[ω[ind_idler-168]./(2e12*pi),ω[ind_idler+168]./(2e12*pi)],linewidth=2,linestyle=:dash,color=RGBf(0.0,0.0,0.0),label="grating \nbandgap")

axislegend(; position=(1,1))
xstart = [Point2f(193.65,(-16.75-(-16.75+0.01)/2)),Point2f(193.65,(-16.75-(-16.75+0.01)/2))]; xdir = [Vec2f(0, -(-16.75+0.01)/2.5), Vec2f(0, (-16.75+0.01)/2.5)]; arrows!(xstart, xdir);#https://discourse.julialang.org/t/two-arrows/107453/2
CairoMakie.text!(position =  CairoMakie.Point2f( 30, 200), "16.7dB", fontsize = 22, space=:pixel)


axes2=Axis(fig[1,2],limits=((nothing,nothing),(-50,-0)),xlabel = "frequency (THz)",xlabelsize=22,ylabelsize=22,xticklabelsize=22,yticklabelsize=22)
CairoMakie.lines!(axes2,ω[g_signal:d_signal]./(2e12*pi),10 .*log10.(abs2.(spect_wg[g_signal:d_signal]./i_f[ind_signal])),color=RGBf(0.0,0.9,1.0),linewidth=3,label="output \nwaveguide");#real c'(z) donne vect onde
CairoMakie.lines!(axes2,ω[g_signal:d_signal]./(2e12*pi),10 .*log10.(abs2.(spect_3dg[g_signal:d_signal]./i_f[ind_signal])),color=RGBf(0.0,0.2,1.0),linewidth=3,label="output \ngrating");#real c'(z) donne vect onde
CairoMakie.scatterlines!(axes2,ω[g_signal:d_signal]./(2e12*pi),10 .*log10.(abs2.(i_f[g_signal:d_signal]./i_f[ind_signal])),color=RGBf(1.0,0.5,0.2),linewidth=2,markersize=2,label="input");#real c'(z) donne vect onde

axislegend(; position=(1,1))

fig


=#



