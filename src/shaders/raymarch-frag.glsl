#version 300 es

precision highp float;
out vec4 out_Col;
in vec2 fs_Pos;

uniform vec2 u_Size;
uniform float u_Frame;

const int MAX_MARCHING_STEPS = 100;
const float MIN_DIST = 0.0;
const float MAX_DIST = 100.0;
const float EPSILON = 0.001;

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

/**
 * Constructive solid geometry intersection operation on SDF-calculated distances.
 */
float intersectSDF(float distA, float distB) {
    return max(distA, distB);
}

/**
 * Constructive solid geometry union operation on SDF-calculated distances.
 */
float unionSDF(float distA, float distB) {
    return min(distA, distB);
}

/**
 * Constructive solid geometry difference operation on SDF-calculated distances.
 */
float differenceSDF(float distA, float distB) {
    return max(distA, -distB);
}

float smin( float a, float b, float k )
{
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

/**
 * Signed distance function for a cube centered at the origin
 * with width = height = length = 2.0
 */
float cubeSDF(vec3 p) {
    // If d.x < 0, then -1 < p.x < 1, and same logic applies to p.y, p.z
    // So if all components of d are negative, then p is inside the unit cube
    vec3 d = abs(p) - vec3(1.0, 1.0, 1.0);
    
    // Assuming p is inside the cube, how far is it from the surface?
    // Result will be negative or zero.
    float insideDistance = min(max(d.x, max(d.y, d.z)), 0.0);
    
    // Assuming p is outside the cube, how far is it from the surface?
    // Result will be positive or zero.
    float outsideDistance = length(max(d, 0.0));
    
    return insideDistance + outsideDistance;
}

/**
 * Signed distance function for a sphere centered at the origin with radius 1.0;
 */
float sphereSDF(vec3 p) {
    return length(p) - 1.0;
}

float sdCappedCylinder( vec3 p, vec2 h )
{
  vec2 d = abs(vec2(length(p.xz),p.y)) - h;
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}

float drip(vec3 p, float h) {
    float capsule = sdCapsule(p, vec3(0., 1., 0.), vec3(0., -h, 0.), 0.07);
    float sphere = sphereSDF((p + vec3(0., h, 0.)) * 10.) / 10.;
    return smin(capsule, sphere, 0.3);
}

float puddle(vec3 p, float s) {
    float sphere = sphereSDF((p + vec3(0., 1./s + 0.5, 0.)) * s) / s;
    return sphere;
}

float animatedDrip(vec3 p, float offset) {
    float t = mod(u_Frame * 0.05 + offset, 5.);
    float y = clamp(-(t * t * t * t), -4., 0.);
    t += 0.4;
    float size = 5. + 10. * clamp(1. - t, 0., 1.);
    float f = sphereSDF((p - vec3(0., y, 0.)) * size) / size;
    return f;
}

float flame(vec3 p) {
    vec3 size = vec3(1. + 0.2 * (sin(u_Frame * 0.5) * sin(u_Frame * 0.7)), 0.5 + 0.1 * (sin(u_Frame * 0.2) * cos(u_Frame * 0.9)), 1. + 0.2 * (cos(u_Frame * 0.2) * cos(u_Frame * 0.5)));
    float f = sphereSDF((p + size.y) * size * 3.2) / (3.2 * length(size));
    return f;
}

float candle(vec3 samplePoint) {
    float sphereDist = sphereSDF((samplePoint + vec3(-1., -4., 0.)) / 3.) * 3.;
    float cylDist = sdCappedCylinder(samplePoint, vec2(1., 3.));
    float candle = cylDist;
    int dripCount = 10;
    int seedCounter = 100;
    for (int i = 0; i < dripCount; i++) {
        float height = 0.5 + rand(vec2(i, seedCounter)) * 2.5;
        seedCounter--;
        float angle = rand(vec2(i, seedCounter)) * 3.1415926 * 1.2 + 2.4;
        seedCounter += i;
        float x = sin(angle);
        float z = cos(angle);
        candle = smin(candle, drip(samplePoint + vec3(x, -1., z), height), 0.03);
    }
    candle = differenceSDF(candle, sphereDist);
    int blobCount = 18;
    for (int i = 0; i < blobCount; i++) {
        float size = 3. + rand(vec2(i, seedCounter)) * 5.;
        seedCounter--;
        float angle = rand(vec2(i, seedCounter)) * 3.1415926 * 2.;
        seedCounter -= i;
        float x = sin(angle);
        float z = cos(angle);
        candle = smin(candle, sphereSDF((samplePoint + vec3(x, -1.4 - 0.4 * sin(angle), z)) * size) / size, 0.2);
    }
    int puddleCount = 4;
    seedCounter = 50;
    for (int i = 0; i < puddleCount; i++) {
        float size = 0.8 + rand(vec2(i, seedCounter)) * 0.2;
        seedCounter--;
        float angle = rand(vec2(i, seedCounter)) * 3.1415926 * 1.2 + 2.4;
        seedCounter += i;
        float x = sin(angle);
        float z = cos(angle);
        candle = smin(candle, puddle(samplePoint + vec3(x, 1.5, z), size), 0.3);
    }
    candle = unionSDF(candle, animatedDrip(samplePoint + vec3(sin(3.), -1.3, cos(3.)), 0.));
    candle = unionSDF(candle, animatedDrip(samplePoint + vec3(sin(-2.6), -1, cos(-2.6)), 0.5));
    candle = unionSDF(candle, animatedDrip(samplePoint + vec3(sin(-1.), -1, cos(-1.)), 2.5));
    float cube = cubeSDF(samplePoint * vec3(0.3, 0.3, 0.3) + vec3(0., 1.7, 0.));
    candle = differenceSDF(candle, cube);
    return candle;
}

float stand(vec3 samplePoint) {
    float stand = differenceSDF(sphereSDF(samplePoint * 0.4 - vec3(0., 0.5, 0.)), cubeSDF(samplePoint * vec3(0.3, 0.3, 0.3) - vec3(0., 1., 0.)));
    stand = unionSDF(stand, sphereSDF(samplePoint * 2. - vec3(0., -3., 0.)));
    stand = unionSDF(stand, sphereSDF((samplePoint - vec3(0., -3.5, 0.)) * vec3(2.5, 0.6, 2.5)) / length(vec3(2.5, 0.6, 2.5))); //unionSDF(stand, sphereSDF((samplePoint - vec3(0., -6., 0.)) * vec3(2.6 - sin(u_Frame * 0.1), 0.3, 2.6 - sin(u_Frame * 0.1))));
    stand = unionSDF(stand, sphereSDF(samplePoint * 1.3 - vec3(0., -7., 0.)));
    stand = unionSDF(stand, sdCappedCylinder(samplePoint - vec3(0., -5.5, 0.), vec2(2., 0.1)));
    return stand;
}

/**
 * Signed distance function describing the scene.
 * 
 * Absolute value of the return value indicates the distance to the surface.
 * Sign indicates whether the point is inside or outside the surface,
 * negative indicating inside.
 */
float sceneSDF(vec3 samplePoint) {
    float candle = candle(samplePoint - vec3(0., 3.3, 0.));
    float stand = stand(samplePoint - vec3(0., 1., 0.));
    return unionSDF(stand, candle);
    // return stand;
}

float udRoundBox( vec3 p, vec3 b, float r )
{
    float c = 9.1;
    p.x = mod(p.x,c)-0.5*c;
    p.z = mod(p.z,c)-0.5*c;
    return length(max(abs(p)-b,0.0))-r;
}

float background(vec3 samplePoint) {
    return udRoundBox(samplePoint - vec3(4., -11.2, 4.), vec3(4., 0.5, 4.), 0.5);
}

/**
 * Return the shortest distance from the eyepoint to the scene surface along
 * the marching direction. If no part of the surface is found between start and end,
 * return end.
 * 
 * eye: the eye point, acting as the origin of the ray
 * marchingDirection: the normalized direction to march in
 * start: the starting distance away from the eye
 * end: the max distance away from the ey to march before giving up
 */
float shortestDistanceToSurface(vec3 eye, vec3 marchingDirection, float start, float end) {
    float depth = start;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = sdCappedCylinder(eye + depth * marchingDirection, vec2(2., 5.5));
        if (dist <= 0.1 + EPSILON) {
            dist = sceneSDF(eye + depth * marchingDirection);
        }
        if (dist < EPSILON) {
			return depth;
        }
        depth += dist;
        if (depth >= end) {
            return end;
        }
    }
    return end;
} 
float shortestDistanceToFlame(vec3 eye, vec3 marchingDirection, float start, float end) {
    float depth = start;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = flame(eye + depth * marchingDirection - vec3(0., 5.5, 0.));
        if (dist < EPSILON * 10.) {
			return depth;
        }
        depth += dist;
        if (depth >= end) {
            return end;
        }
    }
    return end;
}  
float shortestDistanceToBackground(vec3 eye, vec3 marchingDirection, float start, float end) {
    float depth = start;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = background(eye + depth * marchingDirection - vec3(0., 5.5, 0.));
        if (dist < EPSILON * 10.) {
			return depth;
        }
        depth += dist;
        if (depth >= end) {
            return end;
        }
    }
    return end;
}  

/**
 * Return the normalized direction to march in from the eye point for a single pixel.
 * 
 * fieldOfView: vertical field of view in degrees
 * size: resolution of the output image
 * fragCoord: the x,y coordinate of the pixel in the output image
 */
vec3 rayDirection(float fieldOfView, vec2 size, vec2 fragCoord) {
    vec2 xy = fragCoord - size / 2.0;
    float z = size.y / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3(xy, -z));
}

/**
 * Using the gradient of the SDF, estimate the normal on the surface at point p.
 */
vec3 estimateNormal(vec3 p) {
    return normalize(vec3(
        sceneSDF(vec3(p.x + EPSILON, p.y, p.z)) - sceneSDF(vec3(p.x - EPSILON, p.y, p.z)),
        sceneSDF(vec3(p.x, p.y + EPSILON, p.z)) - sceneSDF(vec3(p.x, p.y - EPSILON, p.z)),
        sceneSDF(vec3(p.x, p.y, p.z  + EPSILON)) - sceneSDF(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

vec3 estimateNormalFlame(vec3 p) {
    return normalize(vec3(
        flame(vec3(p.x + EPSILON * 10., p.y, p.z)) - flame(vec3(p.x - EPSILON * 10., p.y, p.z)),
        flame(vec3(p.x, p.y + EPSILON * 10., p.z)) - flame(vec3(p.x, p.y - EPSILON * 10., p.z)),
        flame(vec3(p.x, p.y, p.z  + EPSILON * 10.)) - flame(vec3(p.x, p.y, p.z - EPSILON * 10.))
    ));
}
vec3 estimateNormalBackground(vec3 p) {
    return normalize(vec3(
        background(vec3(p.x + EPSILON * 10., p.y, p.z)) - background(vec3(p.x - EPSILON * 10., p.y, p.z)),
        background(vec3(p.x, p.y + EPSILON * 10., p.z)) - background(vec3(p.x, p.y - EPSILON * 10., p.z)),
        background(vec3(p.x, p.y, p.z  + EPSILON * 10.)) - background(vec3(p.x, p.y, p.z - EPSILON * 10.))
    ));
}

float shadow(vec3 ro, vec3 rd, float mint, float maxt, float k)
{
    float res = 1.0;
    for( float t=mint; t < maxt; )
    {
        float h = sceneSDF(ro + rd*t);
        if( h<0.01 )
            return 0.02;
        res = min( res, k*h/t );
        t += h;
    }
    return res;
}

/**
 * Lighting contribution of a single point light source via Phong illumination.
 * 
 * The vec3 returned is the RGB color of the light's contribution.
 *
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 * lightPos: the position of the light
 * lightIntensity: color/intensity of the light
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
 */
vec3 phongContribForLight(vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye,
                          vec3 lightPos, vec3 lightIntensity) {
    vec3 N = estimateNormal(p);
    vec3 L = normalize(lightPos - p);
    vec3 V = normalize(eye - p);
    vec3 R = normalize(reflect(-L, N));
    
    float dotLN = dot(L, N);
    float dotRV = dot(R, V);
    
    if (dotLN < 0.0) {
        // Light not visible from this point on the surface
        return vec3(0.0, 0.0, 0.0);
    } 
    
    if (dotRV < 0.0) {
        // Light reflection in opposite direction as viewer, apply only diffuse
        // component
        return lightIntensity * (k_d * dotLN);
    }
    return lightIntensity * (k_d * dotLN + k_s * pow(dotRV, alpha));
}

vec3 phongContribForLightBackground(vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye,
                          vec3 lightPos, vec3 lightIntensity) {
    vec3 N = estimateNormalBackground(p);
    vec3 L = normalize(lightPos - p);
    vec3 V = normalize(eye - p);
    vec3 R = normalize(reflect(-L, N));
    
    float dotLN = dot(L, N);
    float dotRV = dot(R, V);
    
    if (dotLN < 0.0) {
        // Light not visible from this point on the surface
        return vec3(0.0, 0.0, 0.0);
    } 
    
    if (dotRV < 0.0) {
        // Light reflection in opposite direction as viewer, apply only diffuse
        // component
        return lightIntensity * (k_d * dotLN);
    }
    return lightIntensity * (k_d * dotLN + k_s * pow(dotRV, alpha));
}

/**
 * Lighting via Phong illumination.
 * 
 * The vec3 returned is the RGB color of that point after lighting is applied.
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
 */
vec3 phongIllumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye) {
    const vec3 ambientLight = 0.5 * vec3(1.0, 1.0, 1.0);
    vec3 color = ambientLight * k_a;
    
    vec3 light1Pos = vec3(0.0,
                          5.0,
                          3.0);
    vec3 light1Intensity = vec3(0.6, 0.6, 0.6);

    vec3 light2Pos = vec3(0.0,
                          5.2 - 0.15 * (sin(u_Frame * 0.2) * cos(u_Frame * 0.9)),
                          0.0);
    vec3 light2Intensity = vec3(0.8, 0.8, 0.6) * (1. - 0.15 * (sin(u_Frame * 0.2) * cos(u_Frame * 0.9)));

    vec3 light3Pos = vec3(0.,
                          3.5,
                          -6.);
    vec3 light3Intensity = vec3(0.4, 0.4, 0.5);
    
    color += phongContribForLight(k_d, k_s, alpha, p, eye,
                                  light1Pos,
                                  light1Intensity)
                                  * shadow(p, normalize(light1Pos - p), 0.1, 10., 8.);

    color += phongContribForLight(k_d, k_s, alpha, p, eye,
                                  light2Pos,
                                  light2Intensity)
                                  * shadow(p, normalize(light2Pos - p), 0.1, 10., 8.);
                                  
    color += phongContribForLight(k_d, k_s, alpha, p, eye,
                                  light3Pos,
                                  light3Intensity)
                                  * shadow(p, normalize(light3Pos - p), 0.1, 10., 8.);

    return color;
}

vec3 phongIlluminationBackground(vec3 k_a, vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye) {
    const vec3 ambientLight = 0.5 * vec3(1.0, 1.0, 1.0);
    vec3 color = ambientLight * k_a;
    
    vec3 light2Pos = vec3(0.0,
                          5.2 - 0.15 * (sin(u_Frame * 0.2) * cos(u_Frame * 0.9)),
                          0.0);
    vec3 light2Intensity = vec3(0.8, 0.8, 0.6) * (1. - 0.15 * (sin(u_Frame * 0.2) * cos(u_Frame * 0.9)));
    
    color += phongContribForLightBackground(k_d, k_s, alpha, p, eye,
                                  light2Pos,
                                  light2Intensity) * shadow(p, normalize(light2Pos - p), 0.1, 10., 8.);

    return color;
}

mat4 viewMatrix(vec3 eye, vec3 center, vec3 up) {
    // Based on gluLookAt man page
    vec3 f = normalize(center - eye);
    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);
    return mat4(
        vec4(s, 0.0),
        vec4(u, 0.0),
        vec4(-f, 0.0),
        vec4(0.0, 0.0, 0.0, 1)
    );
}

vec2 random2( vec2 p ) {
    return normalize(2. * fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453) - 1.);
}

const vec3 amp1 = vec3(0.4, 0.5, 0.8);
const vec3 freq1 = vec3(0.2, 0.4, 0.2);
const vec3 freq2 = vec3(1.0, 1.0, 2.0);
const vec3 amp2 = vec3(0.25, 0.25, 0.0);

const vec3 e = vec3(0.2, 0.5, 0.8);
const vec3 f = vec3(0.2, 0.25, 0.5);
const vec3 g = vec3(1.0, 1.0, 0.1);
const vec3 h = vec3(0.0, 0.8, 0.2);

vec3 Gradient(float t)
{
    return amp1 + freq1 * cos(6.2831 * (freq2 * t + amp2));
}

vec3 Gradient2(float t)
{
    return e + f * cos(6.2831 * (g * t + h));
}

float surflet(vec2 P, vec2 gridPoint)
{
    // Compute falloff function by converting linear distance to amp1 polynomial
    float distX = abs(P.x - gridPoint.x);
    float distY = abs(P.y - gridPoint.y);
    float tX = 1. - 6. * pow(distX, 5.0) + 15. * pow(distX, 4.0) - 10. * pow(distX, 3.0);
    float tY = 1. - 6. * pow(distY, 5.0) + 15. * pow(distY, 4.0) - 10. * pow(distY, 3.0);

    // Get the random vector for the grid point
    vec2 gradient = random2(gridPoint);
    // Get the vector from the grid point to P
    vec2 diff = P - gridPoint;
    // Get the value of our height field by dotting grid->P with our gradient
    float height = dot(diff, gradient);
    // Scale our height field (i.e. reduce it) by our polynomial falloff function
    return height * tX * tY;
}

float PerlinNoise(vec2 uv)
{
    // Tile the space
    vec2 uvXLYL = floor(uv);
    vec2 uvXHYL = uvXLYL + vec2(1.,0.);
    vec2 uvXHYH = uvXLYL + vec2(1.,1.);
    vec2 uvXLYH = uvXLYL + vec2(0.,1.);

    return surflet(uv, uvXLYL) + surflet(uv, uvXHYL) + surflet(uv, uvXHYH) + surflet(uv, uvXLYH);
}

void main()
{
    vec2 coord = vec2(fs_Pos.x * u_Size.x/u_Size.y + 0.5, fs_Pos.y + 0.15);
	vec3 viewDir = rayDirection(20.0, normalize(u_Size), coord);
    vec3 eye = vec3(19.0 + 2.6 * sin(u_Frame * 0.02), 4., 20.0 + 2.6 * -sin(u_Frame * 0.02));
    
    mat4 viewToWorld = viewMatrix(eye, vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0));
    
    vec3 worldDir = (viewToWorld * vec4(viewDir, 0.0)).xyz;
    
    float dist = shortestDistanceToSurface(eye, worldDir, MIN_DIST, MAX_DIST);
    float flameDist = shortestDistanceToFlame(eye, worldDir, MIN_DIST, MAX_DIST);
    float bgDist = shortestDistanceToBackground(eye, worldDir, MIN_DIST, MAX_DIST);
    dist = unionSDF(dist, bgDist);
    vec3 color;
    
    if (dist > MAX_DIST - EPSILON) {
        // Didn't hit anything
        float noise = 0.;
        float n = 4.;
        for (float i = 0.; i < n; i++) {
            noise += PerlinNoise((coord + vec2(rand(vec2(i,i)), rand(vec2(i,i)))) * (4. + 6. * i/n) * vec2(1./coord.y, 1./(coord.y * coord.y))+ vec2(u_Frame * 0.01, 0.)) + 0.5;
        }
        noise /= n;
        noise *= noise;
        noise = clamp(noise, 0., 1.);
        vec3 noiseCol = vec3(noise * 0.5, noise * 0.3, noise * 0.6);
        color = mix(vec3(0.2, 0.2, 0.2), vec3(0.4, 0.4, 0.45), clamp(coord.y*2. + 0.7, 0., 1.));
        color = mix(color, noiseCol, clamp(coord.y*2. - 0.7, 0., 1.));
		if (flameDist > MAX_DIST - EPSILON) {
            out_Col = vec4(color, 1.);
            return;
        }
    }
    
    // The closest point on the surface to the eyepoint along the view ray
    vec3 p = eye + dist * worldDir;
    
    vec3 K_a = 0.9 * vec3(0.3, 0.25, 0.2);
    vec3 K_d = vec3(252./255., 243./255., 209./255.);
    vec3 K_s = vec3(1.0, 1.0, 1.0);
    float shininess = 10.0;

    if (p.y <= -4.6) {
        K_a = 0. * vec3(30./255., 30./255., 30./255.);
        K_d = 0.3 * vec3(81./255., 69./255., 100./255.);
        K_s = vec3(0.8, 0.8, 0.8);
        shininess = 3.;
        color = phongIlluminationBackground(K_a, K_d, K_s, shininess, p, eye);
        color = mix(vec3(0., 0., 0.), color, clamp(p.y * 3. + 15., 0., 1.));
        color = mix(color, vec3(0.4, 0.4, 0.45), clamp(dist/MAX_DIST * dist/MAX_DIST, 0., 1.));
    } 
    else {
        if (p.y <= 1.01) {
            K_a = 0.9 * vec3(30./255., 30./255., 30./255.);
            K_d = vec3(81./255., 69./255., 45./255.);
            K_s = vec3(0.8, 0.8, 0.8);
            shininess = 3.;
        }
        color = phongIllumination(K_a, K_d, K_s, shininess, p, eye);
    }
       

    if (dist > flameDist) {
        vec3 p2 = eye + flameDist * worldDir;
        vec3 N = estimateNormalFlame(p2);
        float d = dot(N, eye - p2);
        d /= 2.;
        d = 1. - clamp(-d*(d-2.), 0., 1.);
        color += vec3(1., 1., 0.6) * d * 0.7;
    }
    
    out_Col = vec4(color, 1.0);
}