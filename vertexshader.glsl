#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 coordinate;

// Output data ; will be interpolated for each fragment.
out vec2 fragmentCoordinate;
out vec3 fragmentNoise;

// Values that stay constant for the whole mesh.
uniform  mat4 MVP ;
uniform float time;


/*
 *  GLOBAL VARIABLES: 
 *  ================  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
#define PI                              3.1415928f
#define FLT_MAX                   3.402823466e+38f
#define EARTH_RADIUS                     6360000.f
#define ATM_TOP_HEIGHT                     80000.f
#define ATM_TOP_RADIUS                     (EARTH_RADIUS+ATM_TOP_HEIGHT)
#define OPTICAL_DEPTH_LUT_DIM              vec4(64,32,64,32)
#define PARTICLE_SCALE_HEIGHT              vec2(7994.f,1200.f)
#define SRF_SCATTERING_IN_PARTICLE_LUT_DIM vec3(32,64,16)
#define VOL_SCATTERING_IN_PARTICLE_LUT_DIM vec4(32,64,32,8)

#define LERP(v0,v1,t) (1-t)*v0 + t*v1
#define SAMPLE_4D_LUT(tex3D, LUT_DIM, f4LUTCoords, Result){   \
    vec3  f3UVW;                                                    \
    f3UVW.xy = f4LUTCoords.xy;                                      \
    float fQSlice = f4LUTCoords.w * LUT_DIM.w - 0.5;                \
    float fQ0Slice = floor(fQSlice);                                \
    float fQWeight = fQSlice - fQ0Slice;                            \
                                                                    \
    f3UVW.z = (fQ0Slice + f4LUTCoords.z) / LUT_DIM.w;               \
                                                                    \
    Result = LERP( texture3D(tex3D,      f3UVW                          ).r,  \ 
                   texture3D(tex3D, frac(f3UVW + vec3(0,0,1/LUT_DIM.w)) ).r,  \ 
                   fQWeight);                                                 \
}

float _fCloudAltitude      = 3000.f;
float _fCloudThickness     =  700.f;
float _fAttenuationCoeff   =  0.07f; // Typical scattering coefficient lies in the range 0.01 - 0.1 m^-1
float _fParticleCutOffDist =  2e+5f;
float _fEarthRadius     = 6360000.f;


/*
 *  GENERAL: 
 *  =======  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

// Pseudorandom number generator
// https://www.geeks3d.com/20100831/shader-library-noise-and-pseudo-random-number-generator-in-glsl/
float rnd(vec2 x){
    int n = int(x.x * 40.0 + x.y * 6400.0);
    n = (n << 13) ^ n;
    return 1.0 - float( (n * (n * n * 15731 + 789221) + \
             1376312589) & 0x7fffffff) / 1073741824.0;
}

// GetRaySphereIntersection
// ........................
// http://wiki.cgsociety.org/index.php/Ray_Sphere_Intersection
#define NO_INTERSECTIONS vec2(-1,-2)

void GetRaySphereIntersection(in  vec3  f3RayOrigin,
                              in  vec3  f3RayDirection,
                              in  vec3  f3SphereCenter,
                              in  float fSphereRadius,
                              out vec2  f2Intersections){
    f3RayOrigin -= f3SphereCenter;
    float A = dot(f3RayDirection, f3RayDirection);
    float B = 2 * dot(f3RayOrigin, f3RayDirection);
    float C = dot(f3RayOrigin,f3RayOrigin) - fSphereRadius*fSphereRadius;
    float D = B*B - 4*A*C;
    // If discriminant is negative, there are no real roots hence the ray misses the
    // sphere
    if( D<0 ){
        f2Intersections = NO_INTERSECTIONS;
    }
    else{
        D = sqrt(D);
        f2Intersections = vec2(-B - D, -B + D) / (2*A); // A must be positive here!!
    }
}



vec3 GetParticleScales(in float fSize){
    vec3 f3Scales = fSize;
    f3Scales.y = min(f3Scales.y, _fCloudThickness/2.f);
    return f3Scales;
}

// This helper function computes intersection of the view ray with the particle ellipsoid
void IntersectRayWithParticle(const in  SParticleAttribs ParticleAttrs,
                              const in  SCloudCellAttribs CellAttrs,
                              const in  vec3 f3CameraPos, 
                              const in  vec3 f3ViewRay,
                              out vec2  f2RayIsecs,
                              out vec3  f3EntryPointUSSpace, // Entry point in Unit Sphere (US) space
                              out vec3  f3ViewRayUSSpace,    // View ray direction in Unit Sphere (US) space
                              out vec3  f3LightDirUSSpace,   // Light direction in Unit Sphere (US) space
                              out float fDistanceToEntryPoint,
                              out float fDistanceToExitPoint){
    // Construct local frame matrix
    vec3 f3Normal    = CellAttrs.f3Normal.xyz;
    vec3 f3Tangent   = CellAttrs.f3Tangent.xyz;
    vec3 f3Bitangent = CellAttrs.f3Bitangent.xyz;
    mat3 f3x3ObjToWorldSpaceRotation = mat3(f3Tangent, f3Normal, f3Bitangent); 
    // World to obj space is inverse of the obj to world space matrix, which is simply transpose
    // for orthogonal matrix:
    mat3 f3x3WorldToObjSpaceRotation = transpose(f3x3ObjToWorldSpaceRotation); 
    
    // Compute camera location and view direction in particle's object space:
    vec3 f3CamPosObjSpace  = f3CameraPos - ParticleAttrs.f3Pos;                         // --------------------------------------------------- ParticleAttrs.f3Pos
         f3CamPosObjSpace  = mul(f3CamPosObjSpace, f3x3WorldToObjSpaceRotation);
    vec3 f3ViewRayObjSpace = mul(f3ViewRay, f3x3WorldToObjSpaceRotation );
    vec3 f3LightDirObjSpce = mul(-g_LightAttribs.f4DirOnLight.xyz, f3x3WorldToObjSpaceRotation ); // ------------------------------------- g_LightAttribs.f4DirOnLight

    // Compute scales to transform ellipsoid into the unit sphere:
    vec3 f3Scale = 1.f / GetParticleScales(ParticleAttrs.fSize);                        //------------------------------------------------- ParticleAttrs.fSize
    
    vec3 f3ScaledCamPosObjSpace;
    f3ScaledCamPosObjSpace  = f3CamPosObjSpace*f3Scale;
    f3ViewRayUSSpace  = normalize(f3ViewRayObjSpace*f3Scale);
    f3LightDirUSSpace = normalize(f3LightDirObjSpce*f3Scale);

    // Scale camera pos and view dir in obj space and compute intersection with the unit sphere:
    GetRaySphereIntersection(f3ScaledCamPosObjSpace, f3ViewRayUSSpace, 0, 1.f, f2RayIsecs);

    f3EntryPointUSSpace = f3ScaledCamPosObjSpace + f3ViewRayUSSpace*f2RayIsecs.x;

    fDistanceToEntryPoint = length(f3ViewRayUSSpace/f3Scale) * f2RayIsecs.x;
    fDistanceToExitPoint  = length(f3ViewRayUSSpace/f3Scale) * f2RayIsecs.y;
}




// This shader renders particles
void RenderCloudsPS(PS_Input In,
                    out float fTransparency : SV_Target0,
                    out float fDistToCloud  : SV_Target1,
                    out vec4  f4Color       : SV_Target2
                    ){

    SParticleAttribs          ParticleAttrs = g_Particles          [In.uiParticleID];
    SCloudCellAttribs           CellAttribs = g_CloudCells         [In.uiParticleID / g_GlobalCloudAttribs.uiMaxLayers];
    SCloudParticleLighting ParticleLighting = g_bufParticleLighting[In.uiParticleID];

    vec3 f3CameraPos, f3ViewRay;

    f3CameraPos = g_CameraAttribs.f4CameraPos.xyz;  // Posicion de camara
    f3ViewRay   =         normalize(In.f3ViewRay);  // direccion de vista?
    
    vec2 f2ScreenDim = vec2(1024, 768);
    vec2 f2PosPS = UVToProj( In.f4Pos.xy / f2ScreenDim );
    
    float fDepth = GetConservativeScreenDepth( ProjToUV(f2PosPS.xy) );
    vec4  f4ReconstructedPosWS = mul( ve4(f2PosPS.xy,fDepth,1.0), g_CameraAttribs.mViewProjInv );
    vec3  f3WorldPos = f4ReconstructedPosWS.xyz / f4ReconstructedPosWS.w;

    // Compute view ray
    f3ViewRay = f3WorldPos - f3CameraPos;
    float fRayLength = length(f3ViewRay);
    f3ViewRay /= fRayLength;

    // Intersect view ray with the particle
    vec2  f2RayIsecs;
    float fDistanceToEntryPoint, fDistanceToExitPoint;
    vec3  f3EntryPointUSSpace, f3ViewRayUSSpace, f3LightDirUSSpace;
    IntersectRayWithParticle(ParticleAttrs, CellAttribs, f3CameraPos,  f3ViewRay,
                             f2RayIsecs, f3EntryPointUSSpace, f3ViewRayUSSpace,
                             f3LightDirUSSpace,
                             fDistanceToEntryPoint, fDistanceToExitPoint);
   
    if( f2RayIsecs.y < 0 || fRayLength < fDistanceToEntryPoint )
        discard;
    fDistanceToExitPoint = min(fDistanceToExitPoint, fRayLength);

    // Compute particle rendering attributes
    ComputeParticleRenderAttribs(   ParticleAttrs, 
                                    CellAttribs,
                                    ParticleLighting,
                                    f3CameraPos,
                                    f3ViewRay,
                                    f3EntryPointUSSpace, 
                                    f3ViewRayUSSpace,
                                    f3LightDirUSSpace,
                                    fDistanceToExitPoint,
                                    fDistanceToEntryPoint,
                                    f4Color
                                );

    fTransparency = f4Color.w;
    f4Color.xyz  *= 1-fTransparency;
    fDistToCloud  = fTransparency < 0.99 ? fDistanceToEntryPoint : +FLT_MAX;
}









/*
 *  Main Funtion
 *  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
void main(){	

	// Output position of the vertex, in clip space : MVP * position
	gl_Position =  MVP * vec4(vertexPosition_modelspace,1);
	fragmentCoordinate = coordinate;

    // Noise
    float n = rnd( gl_Vertex.xyz );
    vec3  f3Noise = vec3(n,n,n);
    fRndAzimuthBias = f3Noise.y+(f3Noise.x-0.5)*time*5e-2;









}

 