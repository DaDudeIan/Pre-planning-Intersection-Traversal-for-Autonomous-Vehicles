#import "../../../lib/mod.typ": *
== Satellite Imagery   <c4:sat>

//#text("ADD BIT ABOUT ZOOM LEVEL. INCLUDE COMPARISON IMAGES", fill: red, weight: "bold")

Satellite imagery is a key component of this thesis project. The imagery will be used for both training and testing the #acr("DL") models, by creating a dataset detailed in @c4:data, and as input to said models during inference. This section covers the acquisition of satellite imagery, the process of signing URLs as required by the #acr("API"), and the code created for these purposes.

This project utilizes Google Maps Static #acr("API") as provided by Google Cloud Platform. The #acr("API") allows for the retrieval of static map images at a given resolution and zoom level. This #acr("API") was chosen due to its ease of use, the quality of the retrieved images, and the fact that it is free to use for a limited number of requests. The #acr("API") is used to retrieve satellite imagery of a given location.

=== Image Acquisition   <c4:sat.acq>

Google Maps Static #acr("API") can retrieve images by forming requests with specific parameters that define the center, zoom level, size, and additional options for the map. For this project, images of type `satellite` are used, as they provide the highest level of detail for each retrieved image. Other types like `roadmap` or `terrain` do not provide enough detail to create a path that would realistically help navigate any kind of intersection as things like line markings are abstracted away.

To request an image, a URL is generated dynamically for the #acr("API"), incorporating the required parameters. The parameters of the #acr("API") request are as follows:
- `center`: The latitude and longitude of the center of the map (e.g. `41.30392`, `-81.90169`).
- `zoom`: The zoom level of the map. 1 is the lowest zoom level, showing the entire Earth, and 21 is the highest zoom level, showing individual buildings.
- `size`: The dimensions of the image to be retrieved, specified in pixels (e.g., `400x400`).
- `maptype`: Specifies the type of map to be retrieved. Options include `roadmap`, `satellite`, `terrain`, and `hybrid`.
- `key`: The #acr("API") key used to authenticate the request.
- `signature`: Secret signing signature given by Google Cloud Platform through their interface.
Furthermore, the #acr("API") allows for markers to be placed on the map, which can be used to highlight specific points of interest. This is, however, not relevant to this project.

==== URL Signing   <c4:sat.acq.url>

While requests to the #acr("API") can be made using only the API key, the usage is severely limited without URL signing. URL signing is a security measure that ensures that requests to the #acr("API") are made by the intended user. The signature is generated using the API key and a secret key provided by Google Cloud Platform. The URL signing algorithm is shown in @alg.url_signing and is provided by Google @url_sign.

#let alg = [
  #algorithm(
    [

      #let ind() = h(2em)
      *Input:* URL, secret \ 
      \ 
      url $<-$ urlparse(URL) \ 
      secret_decoded $<-$ base64_decode(secret) \
      signature $<-$ HMAC_SHA1(secret_decoded, url.path + '?' + url.query) \
      signature $<-$ base64_encode(signature) \
      URL_signed $<-$ URL + '&signature=' + signature \
      \
      *Output:* URL_signed 
    ],
    caption: [URL Signing Algorithm (`sign_url`)]
  )<alg.url_signing>
]

#alg

As input is the URL with filled parameters and the secret key. The algorithm generates a signature using the HMAC-SHA1 algorithm with the secret key and the URL to be signed. The signature is then base64 encoded and appended to the URL as a query parameter. The signed URL can then be used to make requests to the #acr("API").

=== Implementation   <c4:sat.impl>

// detail both retrieval and signing

The main functionality of satellite imagery retrieval can be seen in @code.sat. An example of the output of the functionality can be seen in @fig.sat_example.

#listing([
  ```python
  def get_sat_image(lat: float, lon: float, zoom: int = 18, secret: str = None):
    
    req_url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size=400x400&maptype=satellite&key={API_key}"

    signed_url = sign_url(req_url, secret)
    
    response = requests.get(signed_url)
    return response

    
  def save_sat_image(response: requests.Response, filename: str = "map.png") -> None:
    if response.status_code != 200:
      raise Exception(f"Failed to get image, got status code {response.status_code}")
    
    with open(filename, "wb") as f:
      f.write(response.content)
  ```
],
caption: [Python functions used to retrieve and save satellite imagery (`get_sat_image` and `save_sat_image`)]
) <code.sat>

@code.sat shows two functions, `get_sat_image` and `save_sat_image`, that are used to retrieve and save satellite imagery, respectively. The `get_sat_image` function constructs a URL for the Google Maps Static #acr("API") request and signs it using the `sign_url` function detailed in @alg.url_signing. The signed URL is then used to make a request to the #acr("API"), and the response containing the image is returned. This response can then be passed to the `save_sat_image` function, which saves the image to a file with the specified filename.

// @code.url_signing shows the Python implementation of @alg.url_signing with details following beneath.

// #listing([
//   ```python
//   def sign_url(input_url: str = None, secret: str = None) -> str:
//     if not input_url or not secret:
//         raise Exception("Both input_url and secret are required")

//     url = urlparse.urlparse(input_url)
//     url_to_sign = url.path + "?" + url.query
//     decoded_key = base64.urlsafe_b64decode(secret)
//     signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)
//     encoded_signature = base64.urlsafe_b64encode(signature.digest())
//     original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

//     return original_url + "&signature=" + encoded_signature.decode()
  
//   ```
// ],
// caption: [Python implementation of the URL signing algorithm (`sign_url`)]
// ) <code.url_signing>

// - `sign_url:1`: Function declaration for the URL signing algorithm. It takes in two parameters, `input_url` and `secret`.
// - `sign_url:2-3`: A check is performed to ensure that both `input_url` and `secret` are provided.
// - `sign_url:5`: The provided URL is parsed using the `urlparse` function from the `urlparse` library.
// - `sign_url:6`: The URL to be signed is extracted from the parsed URL object.
// - `sign_url:7`: The secret key is decoded from base64 encoding.
// - `sign_url:8`: The signature is generated using the HMAC-SHA1 algorithm with the decoded secret key and the URL to be signed.
// - `sign_url:9-10`: The resulting signature is then base64 encoded and the URL is reconstructed.
// - `sign_url:12`: The signed URL is returned with the signature appended as a query parameter. An example output could be ```
// https://maps.googleapis.com/maps/api/staticmap?center=41.30392,-81.90169&zoom=18&size=400x400&maptype=satellite&key=<api_key>&signature=<signature>``` with filled `<api_key>` and `<signature>`.

A small `rotate_image` function was also created to rotate the retrieved image by some degrees, as the orientation of the satellite images can vary. The code can be seen in @code:rotate_image. This is meant to help simplify the task performed by the model, as it alleviates the need to handle poorly angled images.

#listing([
  ```python
  def rotate_image(image_path, angle) -> None:
    image_obj = Image.open(image_path)
    rotated_image = image_obj.rotate(angle)
    rotated_image.save(image_path)
  ```
],
caption: [Python function to rotate an image by a specified angle (`rotate_image`)]
) <code:rotate_image>

#let url = [
  #set text(size: 12pt, font: "JetBrainsMono NFM")
  https://maps.googleapis.com/maps/api/staticmap?center=55.780001,9.717275&zoom=18&size=400x400&maptype=satellite&key=wefhuwvjwekrlbvowilerbvkebvlearufhbebwe&signature=aqwhfunojlksdcnipwebfpwebfu=
]

#let fig = {
  image("../../../figures/img/map.png", width: 100%)
}

#std-block(breakable: false)[
  #figure(
  grid(
  columns: (4fr, 3fr),
  column-gutter: 1em,
  align: (horizon, center),
  
  std-block(breakable: false)[
    #box(
      fill: theme.sapphire.darken(80%),
      outset: 0em,
      inset: 0em,
    )
    #url
  ],
  fig
),
caption: [Example of a signed URL and satellite image retrieved using the Google Maps Static API.]
) <fig.sat_example>
]



