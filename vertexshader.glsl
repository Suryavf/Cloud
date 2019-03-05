#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 coordinate;

// Output data ; will be interpolated for each fragment.
out vec3 fragmentColor;
out vec2 fragmentCoordinate;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;


#ifndef OPTIMIZE_SAMPLE_LOCATIONS
#   define OPTIMIZE_SAMPLE_LOCATIONS 1
#endif

#ifndef CORRECT_INSCATTERING_AT_DEPTH_BREAKS
#   define CORRECT_INSCATTERING_AT_DEPTH_BREAKS 0
#endif

//#define SHADOW_MAP_DEPTH_BIAS 1e-4

#ifndef TRAPEZOIDAL_INTEGRATION
#   define TRAPEZOIDAL_INTEGRATION 1
#endif

#ifndef ENABLE_LIGHT_SHAFTS
#   define ENABLE_LIGHT_SHAFTS 1
#endif

#ifndef IS_32BIT_MIN_MAX_MAP
#   define IS_32BIT_MIN_MAX_MAP 0
#endif

#ifndef SINGLE_SCATTERING_MODE
#   define SINGLE_SCATTERING_MODE SINGLE_SCTR_MODE_LUT
#endif

#ifndef MULTIPLE_SCATTERING_MODE
#   define MULTIPLE_SCATTERING_MODE MULTIPLE_SCTR_MODE_OCCLUDED
#endif

#ifndef PRECOMPUTED_SCTR_LUT_DIM
#   define PRECOMPUTED_SCTR_LUT_DIM float4(32,128,32,16)
#endif

#ifndef NUM_RANDOM_SPHERE_SAMPLES
#   define NUM_RANDOM_SPHERE_SAMPLES 128
#endif

#ifndef PERFORM_TONE_MAPPING
#   define PERFORM_TONE_MAPPING 1
#endif

#ifndef LOW_RES_LUMINANCE_MIPS
#   define LOW_RES_LUMINANCE_MIPS 7
#endif

#ifndef TONE_MAPPING_MODE
#   define TONE_MAPPING_MODE TONE_MAPPING_MODE_REINHARD_MOD
#endif

#ifndef LIGHT_ADAPTATION
#   define LIGHT_ADAPTATION 1
#endif

#ifndef SHAFTS_FROM_CLOUDS_MODE
#   define SHAFTS_FROM_CLOUDS_MODE SHAFTS_FROM_CLOUDS_TRANSPARENCY_MAP
#endif

#define INVALID_EPIPOLAR_LINE float4(-1000,-1000, -100, -100)

/*
 *  GENERAL: 
 *  =======  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

// This function for analytical evaluation of particle density integral is 
// provided by Eric Bruneton
// http://www-evasion.inrialpes.fr/Membres/Eric.Bruneton/
//
// optical depth for ray (r,mu) of length d, using analytic formula
// (mu=cos(view zenith angle)), intersections with ground ignored
float2 GetDensityIntegralAnalytic(float r, float mu, float d) {
    float2 f2A = sqrt( (0.5/PARTICLE_SCALE_HEIGHT.xy) * r );
    float4 f4A01 = f2A.xxyy * float2(mu, mu + d / r).xyxy;
    float4 f4A01s = sign(f4A01);
    float4 f4A01sq = f4A01*f4A01;
    
    float2 f2X;
    f2X.x = f4A01s.y > f4A01s.x ? exp(f4A01sq.x) : 0.0;
    f2X.y = f4A01s.w > f4A01s.z ? exp(f4A01sq.z) : 0.0;
    
    float4 f4Y = f4A01s / (2.3193*abs(f4A01) + sqrt(1.52*f4A01sq + 4.0)) * float3(1.0, exp(-d/PARTICLE_SCALE_HEIGHT.xy*(d/(2.0*r)+mu))).xyxz;

    return sqrt((6.2831*PARTICLE_SCALE_HEIGHT)*r) * exp((EARTH_RADIUS-r)/PARTICLE_SCALE_HEIGHT.xy) * (f2X + float2( dot(f4Y.xy, float2(1.0, -1.0)), dot(f4Y.zw, float2(1.0, -1.0)) ));
}

float3 GetExtinctionUnverified(in float3 f3StartPos, in float3 f3EndPos, float3 f3EyeDir, float3 f3EarthCentre){
#if 0
    float2 f2ParticleDensity = IntegrateParticleDensity(f3StartPos, f3EndPos, f3EarthCentre, 20);
#else
    float r = length(f3StartPos-f3EarthCentre);
    float fCosZenithAngle = dot(f3StartPos-f3EarthCentre, f3EyeDir) / r;
    float2 f2ParticleDensity = GetDensityIntegralAnalytic(r, fCosZenithAngle, length(f3StartPos - f3EndPos));
#endif

    // Get optical depth
    float3 f3TotalRlghOpticalDepth = g_MediaParams.f4RayleighExtinctionCoeff.rgb * f2ParticleDensity.x;
    float3 f3TotalMieOpticalDepth  = g_MediaParams.f4MieExtinctionCoeff.rgb * f2ParticleDensity.y;
        
    // Compute extinction
    float3 f3Extinction = exp( -(f3TotalRlghOpticalDepth + f3TotalMieOpticalDepth) );
    return f3Extinction;
}

float3 GetExtinction(in float3 f3StartPos, in float3 f3EyeDir, in float fRayLength){
    float3 f3EarthCentre = /*g_CameraAttribs.f4CameraPos.xyz*float3(1,0,1)*/ - float3(0,1,0) * EARTH_RADIUS;

    float2 f2RayAtmTopIsecs; 
    // Compute intersections of the view ray with the atmosphere
    GetRaySphereIntersection(f3StartPos, f3EyeDir, f3EarthCentre, ATM_TOP_RADIUS, f2RayAtmTopIsecs);
    // If the ray misses the atmosphere, there is no extinction
    if( f2RayAtmTopIsecs.y < 0 )return 1;

    // Do not let the start and end point be outside the atmosphere
    float3 f3EndPos = f3StartPos + f3EyeDir * min(f2RayAtmTopIsecs.y, fRayLength);
    f3StartPos += f3EyeDir * max(f2RayAtmTopIsecs.x, 0);

    return GetExtinctionUnverified(f3StartPos, f3EndPos, f3EyeDir, f3EarthCentre);
}

float3 GetExtinction(in float3 f3StartPos, in float3 f3EndPos){
    float3 f3EyeDir = f3EndPos - f3StartPos;
    float fRayLength = length(f3EyeDir);
    f3EyeDir /= fRayLength;

    return GetExtinction(f3StartPos, f3EyeDir, fRayLength);
}

float GetCosHorizonAnlge(float fHeight){
    // Due to numeric precision issues, fHeight might sometimes be slightly negative
    fHeight = max(fHeight, 0);
    return -sqrt(fHeight * (2*EARTH_RADIUS + fHeight) ) / (EARTH_RADIUS + fHeight);
}


float ZenithAngle2TexCoord(float fCosZenithAngle, float fHeight, in float fTexDim, float power, float fPrevTexCoord){
    fCosZenithAngle = fCosZenithAngle;
    float fTexCoord;
    float fCosHorzAngle = GetCosHorizonAnlge(fHeight);
    // When performing look-ups into the scattering texture, it is very important that all the look-ups are consistent
    // wrt to the horizon. This means that if the first look-up is above (below) horizon, then the second look-up
    // should also be above (below) horizon. 
    // We use previous texture coordinate, if it is provided, to find out if previous look-up was above or below
    // horizon. If texture coordinate is negative, then this is the first look-up
    bool bIsAboveHorizon = fPrevTexCoord >= 0.5;
    bool bIsBelowHorizon = 0 <= fPrevTexCoord && fPrevTexCoord < 0.5;
    if(  bIsAboveHorizon || !bIsBelowHorizon && (fCosZenithAngle > fCosHorzAngle) ){
        // Scale to [0,1]
        fTexCoord = saturate( (fCosZenithAngle - fCosHorzAngle) / (1 - fCosHorzAngle) );
        fTexCoord = pow(fTexCoord, power);
        // Now remap texture coordinate to the upper half of the texture.
        // To avoid filtering across discontinuity at 0.5, we must map
        // the texture coordinate to [0.5 + 0.5/fTexDim, 1 - 0.5/fTexDim]
        //
        //      0.5   1.5               D/2+0.5        D-0.5  texture coordinate x dimension
        //       |     |                   |            |
        //    |  X  |  X  | .... |  X  ||  X  | .... |  X  |  
        //       0     1          D/2-1   D/2          D-1    texel index
        //
        fTexCoord = 0.5f + 0.5f / fTexDim + fTexCoord * (fTexDim/2 - 1) / fTexDim;
    }
    else{
        fTexCoord = saturate( (fCosHorzAngle - fCosZenithAngle) / (fCosHorzAngle - (-1)) );
        fTexCoord = pow(fTexCoord, power);
        // Now remap texture coordinate to the lower half of the texture.
        // To avoid filtering across discontinuity at 0.5, we must map
        // the texture coordinate to [0.5, 0.5 - 0.5/fTexDim]
        //
        //      0.5   1.5        D/2-0.5             texture coordinate x dimension
        //       |     |            |       
        //    |  X  |  X  | .... |  X  ||  X  | .... 
        //       0     1          D/2-1   D/2        texel index
        //
        fTexCoord = 0.5f / fTexDim + fTexCoord * (fTexDim/2 - 1) / fTexDim;
    }    

    return fTexCoord;
}





float4 WorldParams2InsctrLUTCoords(float fHeight,
                                   float fCosViewZenithAngle,
                                   float fCosSunZenithAngle,
                                   float fCosSunViewAngle,
                                   in float4 f4RefUVWQ){
    float4 f4UVWQ;

    // Limit allowable height range to [SafetyHeightMargin, AtmTopHeight - SafetyHeightMargin] to
    // avoid numeric issues at the Earth surface and the top of the atmosphere
    // (ray/Earth and ray/top of the atmosphere intersection tests are unstable when fHeight == 0 and
    // fHeight == AtmTopHeight respectively)
    fHeight = clamp(fHeight, SafetyHeightMargin, g_MediaParams.fAtmTopHeight - SafetyHeightMargin);
    f4UVWQ.x = saturate( (fHeight - SafetyHeightMargin) / (g_MediaParams.fAtmTopHeight - 2*SafetyHeightMargin) );

#if NON_LINEAR_PARAMETERIZATION
    f4UVWQ.x = pow(f4UVWQ.x, HeightPower);

    f4UVWQ.y = ZenithAngle2TexCoord(fCosViewZenithAngle, fHeight, PRECOMPUTED_SCTR_LUT_DIM.y, ViewZenithPower, f4RefUVWQ.y);
    
    // Use Eric Bruneton's formula for cosine of the sun-zenith angle
    f4UVWQ.z = (atan(max(fCosSunZenithAngle, -0.1975) * tan(1.26 * 1.1)) / 1.1 + (1.0 - 0.26)) * 0.5;

    fCosSunViewAngle = clamp(fCosSunViewAngle, -1, +1);
    f4UVWQ.w = acos(fCosSunViewAngle) / PI;
    f4UVWQ.w = sign(f4UVWQ.w - 0.5) * pow( abs((f4UVWQ.w - 0.5)/0.5), SunViewPower)/2 + 0.5;
    
    f4UVWQ.xzw = ((f4UVWQ * (PRECOMPUTED_SCTR_LUT_DIM-1) + 0.5) / PRECOMPUTED_SCTR_LUT_DIM).xzw;
#else
    f4UVWQ.y = (fCosViewZenithAngle+1.f) / 2.f;
    f4UVWQ.z = (fCosSunZenithAngle +1.f) / 2.f;
    f4UVWQ.w = (fCosSunViewAngle   +1.f) / 2.f;

    f4UVWQ = (f4UVWQ * (PRECOMPUTED_SCTR_LUT_DIM-1) + 0.5) / PRECOMPUTED_SCTR_LUT_DIM;
#endif

    return f4UVWQ;
}

float3 LookUpPrecomputedScattering(float3 f3StartPoint, 
                                   float3 f3ViewDir, 
                                   float3 f3EarthCentre,
                                   float3 f3DirOnLight,
                                   in Texture3D<float3> tex3DScatteringLUT,
                                   inout float4 f4UVWQ){
    float3 f3EarthCentreToPointDir = f3StartPoint - f3EarthCentre;
    float fDistToEarthCentre = length(f3EarthCentreToPointDir);
    f3EarthCentreToPointDir /= fDistToEarthCentre;
    float fHeightAboveSurface = fDistToEarthCentre - EARTH_RADIUS;
    float fCosViewZenithAngle = dot( f3EarthCentreToPointDir, f3ViewDir    );
    float fCosSunZenithAngle  = dot( f3EarthCentreToPointDir, f3DirOnLight );
    float fCosSunViewAngle    = dot( f3ViewDir,               f3DirOnLight );

    // Provide previous look-up coordinates
    f4UVWQ = WorldParams2InsctrLUTCoords(fHeightAboveSurface, fCosViewZenithAngle,
                                         fCosSunZenithAngle, fCosSunViewAngle, 
                                         f4UVWQ);

    float3 f3UVW0; 
    f3UVW0.xy = f4UVWQ.xy;
    float fQ0Slice = floor(f4UVWQ.w * PRECOMPUTED_SCTR_LUT_DIM.w - 0.5);
    fQ0Slice = clamp(fQ0Slice, 0, PRECOMPUTED_SCTR_LUT_DIM.w-1);
    float fQWeight = (f4UVWQ.w * PRECOMPUTED_SCTR_LUT_DIM.w - 0.5) - fQ0Slice;
    fQWeight = max(fQWeight, 0);
    float2 f2SliceMinMaxZ = float2(fQ0Slice, fQ0Slice+1)/PRECOMPUTED_SCTR_LUT_DIM.w + float2(0.5,-0.5) / (PRECOMPUTED_SCTR_LUT_DIM.z*PRECOMPUTED_SCTR_LUT_DIM.w);
    f3UVW0.z =  (fQ0Slice + f4UVWQ.z) / PRECOMPUTED_SCTR_LUT_DIM.w;
    f3UVW0.z = clamp(f3UVW0.z, f2SliceMinMaxZ.x, f2SliceMinMaxZ.y);
    
    float fQ1Slice = min(fQ0Slice+1, PRECOMPUTED_SCTR_LUT_DIM.w-1);
    float fNextSliceOffset = (fQ1Slice - fQ0Slice) / PRECOMPUTED_SCTR_LUT_DIM.w;
    float3 f3UVW1 = f3UVW0 + float3(0,0,fNextSliceOffset);
    float3 f3Insctr0 = tex3DScatteringLUT.SampleLevel(samLinearClamp, f3UVW0, 0);
    float3 f3Insctr1 = tex3DScatteringLUT.SampleLevel(samLinearClamp, f3UVW1, 0);
    float3 f3Inscattering = lerp(f3Insctr0, f3Insctr1, fQWeight);

    return f3Inscattering;
}













float TexCoord2ZenithAngle(float fTexCoord, float fHeight, in float fTexDim, float power){
    float fCosZenithAngle;

    float fCosHorzAngle = GetCosHorizonAnlge(fHeight);
    if( fTexCoord > 0.5 )
    {
        // Remap to [0,1] from the upper half of the texture [0.5 + 0.5/fTexDim, 1 - 0.5/fTexDim]
        fTexCoord = saturate( (fTexCoord - (0.5f + 0.5f / fTexDim)) * fTexDim / (fTexDim/2 - 1) );
        fTexCoord = pow(fTexCoord, 1/power);
        // Assure that the ray does NOT hit Earth
        fCosZenithAngle = max( (fCosHorzAngle + fTexCoord * (1 - fCosHorzAngle)), fCosHorzAngle + 1e-4);
    }
    else
    {
        // Remap to [0,1] from the lower half of the texture [0.5, 0.5 - 0.5/fTexDim]
        fTexCoord = saturate((fTexCoord - 0.5f / fTexDim) * fTexDim / (fTexDim/2 - 1));
        fTexCoord = pow(fTexCoord, 1/power);
        // Assure that the ray DOES hit Earth
        fCosZenithAngle = min( (fCosHorzAngle - fTexCoord * (fCosHorzAngle - (-1))), fCosHorzAngle - 1e-4);
    }
    return fCosZenithAngle;
}

static const float SafetyHeightMargin = 16.f;
#define NON_LINEAR_PARAMETERIZATION 1
static const float HeightPower = 0.5f;
static const float ViewZenithPower = 0.2;
static const float SunViewPower = 1.5f;

void InsctrLUTCoords2WorldParams(in float4 f4UVWQ,
                                 out float fHeight,
                                 out float fCosViewZenithAngle,
                                 out float fCosSunZenithAngle,
                                 out float fCosSunViewAngle){
#if NON_LINEAR_PARAMETERIZATION
    // Rescale to exactly 0,1 range
    f4UVWQ.xzw = saturate((f4UVWQ* PRECOMPUTED_SCTR_LUT_DIM - 0.5) / (PRECOMPUTED_SCTR_LUT_DIM-1)).xzw;

    f4UVWQ.x = pow( f4UVWQ.x, 1/HeightPower );
    // Allowable height range is limited to [SafetyHeightMargin, AtmTopHeight - SafetyHeightMargin] to
    // avoid numeric issues at the Earth surface and the top of the atmosphere
    fHeight = f4UVWQ.x * (g_MediaParams.fAtmTopHeight - 2*SafetyHeightMargin) + SafetyHeightMargin;

    fCosViewZenithAngle = TexCoord2ZenithAngle(f4UVWQ.y, fHeight, PRECOMPUTED_SCTR_LUT_DIM.y, ViewZenithPower);
    
    // Use Eric Bruneton's formula for cosine of the sun-zenith angle
    fCosSunZenithAngle = tan((2.0 * f4UVWQ.z - 1.0 + 0.26) * 1.1) / tan(1.26 * 1.1);

    f4UVWQ.w = sign(f4UVWQ.w - 0.5) * pow( abs((f4UVWQ.w - 0.5)*2), 1/SunViewPower)/2 + 0.5;
    fCosSunViewAngle = cos(f4UVWQ.w*PI);
#else
    // Rescale to exactly 0,1 range
    f4UVWQ = (f4UVWQ * PRECOMPUTED_SCTR_LUT_DIM - 0.5) / (PRECOMPUTED_SCTR_LUT_DIM-1);

    // Allowable height range is limited to [SafetyHeightMargin, AtmTopHeight - SafetyHeightMargin] to
    // avoid numeric issues at the Earth surface and the top of the atmosphere
    fHeight = f4UVWQ.x * (g_MediaParams.fAtmTopHeight - 2*SafetyHeightMargin) + SafetyHeightMargin;

    fCosViewZenithAngle = f4UVWQ.y * 2 - 1;
    fCosSunZenithAngle  = f4UVWQ.z * 2 - 1;
    fCosSunViewAngle    = f4UVWQ.w * 2 - 1;
#endif

    fCosViewZenithAngle = clamp(fCosViewZenithAngle, -1, +1);
    fCosSunZenithAngle  = clamp(fCosSunZenithAngle,  -1, +1);
    // Compute allowable range for the cosine of the sun view angle for the given
    // view zenith and sun zenith angles
    float D = (1.0 - fCosViewZenithAngle * fCosViewZenithAngle) * (1.0 - fCosSunZenithAngle  * fCosSunZenithAngle);
    
    // !!!!  IMPORTANT NOTE regarding NVIDIA hardware !!!!

    // There is a very weird issue on NVIDIA hardware with clamp(), saturate() and min()/max() 
    // functions. No matter what function is used, fCosViewZenithAngle and fCosSunZenithAngle
    // can slightly fall outside [-1,+1] range causing D to be negative
    // Using saturate(D), max(D, 0) and even D>0?D:0 does not work!
    // The only way to avoid taking the square root of negative value and obtaining NaN is 
    // to use max() with small positive value:
    D = sqrt( max(D, 1e-20) );
    
    float2 f2MinMaxCosSunViewAngle = fCosViewZenithAngle*fCosSunZenithAngle + float2(-D, +D);
    
	// Clamp to allowable range
    fCosSunViewAngle    = clamp(fCosSunViewAngle, f2MinMaxCosSunViewAngle.x, f2MinMaxCosSunViewAngle.y);
}

























float3 ComputeViewDir(in float fCosViewZenithAngle){
    return float3(sqrt(saturate(1 - fCosViewZenithAngle*fCosViewZenithAngle)), fCosViewZenithAngle, 0);
}

float3 ComputeLightDir(in float3 f3ViewDir, in float fCosSunZenithAngle, in float fCosSunViewAngle){
    float3 f3DirOnLight;

    f3DirOnLight.x = (f3ViewDir.x > 0) ? (fCosSunViewAngle - fCosSunZenithAngle * f3ViewDir.y) / f3ViewDir.x : 0;
    f3DirOnLight.y = fCosSunZenithAngle;
    f3DirOnLight.z = sqrt( saturate(1 - dot(f3DirOnLight.xy, f3DirOnLight.xy)) );
    
	// Do not normalize f3DirOnLight! Even if its length is not exactly 1 (which can 
    // happen because of fp precision issues), all the dot products will still be as 
    // specified, which is essentially important. If we normalize the vector, all the 
    // dot products will deviate, resulting in wrong pre-computation.
    // Since fCosSunViewAngle is clamped to allowable range, f3DirOnLight should always
    // be normalized. However, due to some issues on NVidia hardware sometimes
    // it may not be as that (see IMPORTANT NOTE regarding NVIDIA hardware)
    //f3DirOnLight = normalize(f3DirOnLight);
    return f3DirOnLight;
}



/*
 *  AMBIENT SKY LIGHT: 
 *  =================  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

float3 PrecomputeAmbientSkyLightPS(SScreenSizeQuadVSOutput In) : SV_Target{
    float fU = ProjToUV(In.m_f2PosPS).x;
    float3 f3RayStart = float3(0,20,0);
    float3 f3EarthCentre =  -float3(0,1,0) * EARTH_RADIUS;
    float fCosZenithAngle = clamp(fU * 2 - 1, -1, +1);
    float3 f3DirOnLight = float3(sqrt(saturate(1 - fCosZenithAngle*fCosZenithAngle)), fCosZenithAngle, 0);
    float3 f3SkyLight = 0;
    // Go through a number of random directions on the sphere
    for(int iSample = 0; iSample < NUM_RANDOM_SPHERE_SAMPLES; ++iSample){
        // Get random direction
        float3 f3RandomDir = normalize( g_tex2DSphereRandomSampling.Load(int3(iSample,0,0)) );
        // Reflect directions from the lower hemisphere
        f3RandomDir.y = abs(f3RandomDir.y);
        // Get multiple scattered light radiance when looking in direction f3RandomDir (the light thus goes in direction -f3RandomDir)
        float4 f4UVWQ = -1;
        float3 f3Sctr = LookUpPrecomputedScattering(f3RayStart, f3RandomDir, f3EarthCentre, f3DirOnLight.xyz, g_tex3DPreviousSctrOrder, f4UVWQ);  //------------------------------------------------
        // Accumulate ambient irradiance through the horizontal plane
        f3SkyLight += f3Sctr * dot(f3RandomDir, float3(0,1,0));
    }
    // Each sample covers 2 * PI / NUM_RANDOM_SPHERE_SAMPLES solid angle (integration is performed over
    // upper hemisphere)
    return f3SkyLight * 2 * PI / NUM_RANDOM_SPHERE_SAMPLES;
}



/*
 *  OPTICAL DEPTH: 
 *  =============  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
 





/*
 *  SINGLE SCATTERING: 
 *  =================  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */



// This shader pre-computes the radiance of single scattering at a given point in given
// direction.
float3 PrecomputeSingleScatteringPS(SScreenSizeQuadVSOutput In) : SV_Target{
    // Get attributes for the current point
    float2 f2UV = ProjToUV(In.m_f2PosPS);
    float fHeight, fCosViewZenithAngle, fCosSunZenithAngle, fCosSunViewAngle;
    InsctrLUTCoords2WorldParams(float4(f2UV, g_MiscParams.f2WQ), fHeight, fCosViewZenithAngle, fCosSunZenithAngle, fCosSunViewAngle );
    float3 f3EarthCentre =  - float3(0,1,0) * EARTH_RADIUS;
    float3 f3RayStart = float3(0, fHeight, 0);
    float3 f3ViewDir = ComputeViewDir(fCosViewZenithAngle);
    float3 f3DirOnLight = ComputeLightDir(f3ViewDir, fCosSunZenithAngle, fCosSunViewAngle);
  
    // Intersect view ray with the top of the atmosphere and the Earth
    float4 f4Isecs;
    GetRaySphereIntersection2( f3RayStart, f3ViewDir, f3EarthCentre, 
                               float2(EARTH_RADIUS, ATM_TOP_RADIUS), 
                               f4Isecs);
    float2 f2RayEarthIsecs  = f4Isecs.xy;
    float2 f2RayAtmTopIsecs = f4Isecs.zw;

    if(f2RayAtmTopIsecs.y <= 0)
        return 0; // This is just a sanity check and should never happen
                  // as the start point is always under the top of the 
                  // atmosphere (look at InsctrLUTCoords2WorldParams())

    // Set the ray length to the distance to the top of the atmosphere
    float fRayLength = f2RayAtmTopIsecs.y;
    // If ray hits Earth, limit the length by the distance to the surface
    if(f2RayEarthIsecs.x > 0)
        fRayLength = min(fRayLength, f2RayEarthIsecs.x);
    
    float3 f3RayEnd = f3RayStart + f3ViewDir * fRayLength;

    float fCloudTransparency = 1;
    float fDitToCloud = +FLT_MAX;
    // Integrate single-scattering
    float3 f3Inscattering, f3Extinction;
    IntegrateUnshadowedInscattering(f3RayStart, 
                                    f3RayEnd,
                                    f3ViewDir,
                                    f3EarthCentre,
                                    f3DirOnLight.xyz,
                                    100,
                                    f3Inscattering,
                                    f3Extinction,
                                    fCloudTransparency,
                                    fDitToCloud);

    return f3Inscattering;
}




/*
 *  MULTIPLE SCATTERING: 
 *  ===================  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

// This shader pre-computes the radiance of light scattered at a given point in given
// direction. It multiplies the previous order in-scattered light with the phase function 
// for each type of particles and integrates the result over the whole set of directions,
// see eq. (7) in [Bruneton and Neyret 08].
float3 ComputeSctrRadiancePS(SScreenSizeQuadVSOutput In) : SV_Target{
    // Get attributes for the current point
    float2 f2UV = ProjToUV(In.m_f2PosPS);
    float fHeight, fCosViewZenithAngle, fCosSunZenithAngle, fCosSunViewAngle;
    InsctrLUTCoords2WorldParams( float4(f2UV, g_MiscParams.f2WQ), fHeight, fCosViewZenithAngle, fCosSunZenithAngle, fCosSunViewAngle );
    float3 f3EarthCentre =  - float3(0,1,0) * EARTH_RADIUS;
    float3 f3RayStart = float3(0, fHeight, 0);
    float3 f3ViewDir = ComputeViewDir(fCosViewZenithAngle);
    float3 f3DirOnLight = ComputeLightDir(f3ViewDir, fCosSunZenithAngle, fCosSunViewAngle);
    
    // Compute particle density scale factor
    float2 f2ParticleDensity = exp( -fHeight / PARTICLE_SCALE_HEIGHT );
    
    float3 f3SctrRadiance = 0;
    // Go through a number of samples randomly distributed over the sphere
    for(int iSample = 0; iSample < NUM_RANDOM_SPHERE_SAMPLES; ++iSample){
        // Get random direction
        float3 f3RandomDir = normalize( g_tex2DSphereRandomSampling.Load(int3(iSample,0,0)) );
        // Get the previous order in-scattered light when looking in direction f3RandomDir (the light thus goes in direction -f3RandomDir)
        float4 f4UVWQ = -1;
        float3 f3PrevOrderSctr = LookUpPrecomputedScattering(f3RayStart, f3RandomDir, f3EarthCentre, f3DirOnLight.xyz, g_tex3DPreviousSctrOrder, f4UVWQ); 
        
        // Apply phase functions for each type of particles
        // Note that total scattering coefficients are baked into the angular scattering coeffs
        float3 f3DRlghInsctr = f2ParticleDensity.x * f3PrevOrderSctr;
        float3 f3DMieInsctr  = f2ParticleDensity.y * f3PrevOrderSctr;
        float fCosTheta = dot(f3ViewDir, f3RandomDir);
        ApplyPhaseFunctions(f3DRlghInsctr, f3DMieInsctr, fCosTheta);

        f3SctrRadiance += f3DRlghInsctr + f3DMieInsctr;
    }
    // Since we tested N random samples, each sample covered 4*Pi / N solid angle
    // Note that our phase function is normalized to 1 over the sphere. For instance,
    // uniform phase function would be p(theta) = 1 / (4*Pi).
    // Notice that for uniform intensity I if we get N samples, we must obtain exactly I after
    // numeric integration
    return f3SctrRadiance * 4*PI / NUM_RANDOM_SPHERE_SAMPLES;
}

// This shader computes in-scattering order for a given point and direction. It performs integration of the 
// light scattered at particular point along the ray, see eq. (11) in [Bruneton and Neyret 08].
float3 ComputeScatteringOrderPS(SScreenSizeQuadVSOutput In) : SV_Target{
    // Get attributes for the current point
    float2 f2UV = ProjToUV(In.m_f2PosPS);
    float fHeight, fCosViewZenithAngle, fCosSunZenithAngle, fCosSunViewAngle;
    InsctrLUTCoords2WorldParams(float4(f2UV, g_MiscParams.f2WQ), fHeight, fCosViewZenithAngle, fCosSunZenithAngle, fCosSunViewAngle );
    float3 f3EarthCentre =  - float3(0,1,0) * EARTH_RADIUS;
    float3 f3RayStart = float3(0, fHeight, 0);
    float3 f3ViewDir = ComputeViewDir(fCosViewZenithAngle);
    float3 f3DirOnLight = ComputeLightDir(f3ViewDir, fCosSunZenithAngle, fCosSunViewAngle);
    
    // Intersect the ray with the atmosphere and Earth
    float4 f4Isecs;
    GetRaySphereIntersection2( f3RayStart, f3ViewDir, f3EarthCentre, 
                               float2(EARTH_RADIUS, ATM_TOP_RADIUS), 
                               f4Isecs);
    float2 f2RayEarthIsecs  = f4Isecs.xy;
    float2 f2RayAtmTopIsecs = f4Isecs.zw;

    if(f2RayAtmTopIsecs.y <= 0)
        return 0; // This is just a sanity check and should never happen
                  // as the start point is always under the top of the 
                  // atmosphere (look at InsctrLUTCoords2WorldParams())

    float fRayLength = f2RayAtmTopIsecs.y;
    if(f2RayEarthIsecs.x > 0)
        fRayLength = min(fRayLength, f2RayEarthIsecs.x);
    
    float3 f3RayEnd = f3RayStart + f3ViewDir * fRayLength;

    const float fNumSamples = 64;
    float fStepLen = fRayLength / fNumSamples;

    float4 f4UVWQ = -1;
    float3 f3PrevSctrRadiance = LookUpPrecomputedScattering(f3RayStart, f3ViewDir, f3EarthCentre, f3DirOnLight.xyz, g_tex3DPointwiseSctrRadiance, f4UVWQ); 
    float2 f2PrevParticleDensity = exp( -fHeight / PARTICLE_SCALE_HEIGHT );

    float2 f2NetParticleDensityFromCam = 0;
    float3 f3Inscattering = 0;

    for(float fSample=1; fSample <= fNumSamples; ++fSample){
        float3 f3Pos = lerp(f3RayStart, f3RayEnd, fSample/fNumSamples);

        float fCurrHeight = length(f3Pos - f3EarthCentre) - EARTH_RADIUS;
        float2 f2ParticleDensity = exp( -fCurrHeight / PARTICLE_SCALE_HEIGHT );

        f2NetParticleDensityFromCam += (f2PrevParticleDensity + f2ParticleDensity) * (fStepLen / 2.f);
        f2PrevParticleDensity = f2ParticleDensity;
        
        // Get optical depth
        float3 f3RlghOpticalDepth = g_MediaParams.f4RayleighExtinctionCoeff.rgb * f2NetParticleDensityFromCam.x;
        float3 f3MieOpticalDepth  = g_MediaParams.f4MieExtinctionCoeff.rgb      * f2NetParticleDensityFromCam.y;
        
        // Compute extinction from the camera for the current integration point:
        float3 f3ExtinctionFromCam = exp( -(f3RlghOpticalDepth + f3MieOpticalDepth) );

        // Get attenuated scattered light radiance in the current point
        float4 f4UVWQ = -1;
        float3 f3SctrRadiance = f3ExtinctionFromCam * LookUpPrecomputedScattering(f3Pos, f3ViewDir, f3EarthCentre, f3DirOnLight.xyz, g_tex3DPointwiseSctrRadiance, f4UVWQ); 
        // Update in-scattering integral
        f3Inscattering += (f3SctrRadiance +  f3PrevSctrRadiance) * (fStepLen/2.f);
        f3PrevSctrRadiance = f3SctrRadiance;
    }

    return f3Inscattering;
}

float3 AddScatteringOrderPS(SScreenSizeQuadVSOutput In) : SV_Target{
    // Accumulate in-scattering using alpha-blending
    return g_tex3DPreviousSctrOrder.Load( uint4(In.m_f4Pos.xy, g_MiscParams.uiDepthSlice, 0) );
}






/*
 *  Main Funtion
 *  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
void main(){	

	// Output position of the vertex, in clip space : MVP * position
	gl_Position =  MVP * vec4(vertexPosition_modelspace,1);

	// The color of each vertex will be interpolated
	// to produce the color of each fragment
	fragmentColor = vec3(0.5,0.5,0.5);//vertexColor;
	fragmentCoordinate = coordinate;
}

 