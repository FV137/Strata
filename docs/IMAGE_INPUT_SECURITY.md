# Image input security

Image requests upload bytes by default. Server-local paths, `file:` URLs and plain HTTP URLs are rejected.
Terminal `/image <path>` still works: the chat client reads the picture and sends a base64 data URL.
OpenAI image parts and Anthropic base64 image blocks use the same validation path.

## Optional remote images

If a client needs URL images, add exact HTTPS origins to the existing `vision` section of the server config:

```json
{
  "vision": {
    "image_origins": ["https://images.example.com", "https://photos.example.com:8443"]
  }
}
```

Keep the existing `exe`, `model`, `mmproj` and other vision settings. Restart the server after editing. Setup
preserves this setting when reinstalling. An omitted or empty list disables remote images; malformed entries
reject vision startup. There are no wildcard hosts, embedded credentials, URL paths, fragments or query strings
in the origin list. Use ASCII hostnames (punycode for internationalized names).

Every fetch checks the exact host and port before DNS. All returned addresses must be public unicast addresses;
loopback, private, link-local, shared, reserved and transition addresses are rejected. The TCP connection uses one
of those checked literal addresses, with no second DNS lookup. TLS verifies the original hostname and certificate,
and the HTTP Host header retains that hostname. Environment proxy settings and ambient credentials are not used.

Redirects are rejected. Configure the final origin and send the final URL. This applies even to a redirect to a
second allowed origin. Responses must be HTTP 200 with an image Content-Type. HTTP content compression and
ambiguous length/transfer headers are rejected. Chunked and close-delimited bodies remain bounded.

Choosing an allowed origin gives it the image URL (including any query), the server's source IP and request timing.
Use origins you trust and avoid credentials in query strings. The policy is shared by every API-key holder; it is
not a separate policy per user. It does not restrict the configured vision executable or administrator-controlled
model/projector paths.

## Limits and decoding

- At most 16 images per request, checked before tokenization or image encoding.
- At most 16 MiB of downloaded or decoded base64 bytes per image, and 16 MiB after PNG normalization.
- At most 16 million pixels and 16,000 pixels per dimension, checked before decoding image pixels.
- A 15-second network deadline after DNS, including TLS, response headers and body. The system resolver's own
  timeout applies to DNS; this change does not impose a separate resolver deadline.
- The existing 32 MiB API request-body limit still applies, including base64 overhead and other message data.

Pillow validates every accepted raster format: JPEG, PNG, BMP, GIF, WebP, TIFF and AVIF. No magic-byte shortcut
passes input straight to the native encoder, and formats that can invoke external converters (such as EPS) are
excluded. Only the first animation frame is used. A fresh RGB PNG removes source metadata and flattens transparent
pixels to white before native encoding. Pillow is required even for PNG/JPEG inputs.

These checks reduce file-access, SSRF and resource risks. They are not an OS sandbox for Pillow or the native
encoder, a memory-safety audit, a limit on concurrent clients, or a CPU deadline for image decoding. Keep codecs
updated and retain the API authentication/bind protections.

## Verification

Run `python -m unittest serve.test_image_security -v` from the repository root. Tests generate their own images
and run local HTTP/TLS fixtures; no model, GPU or external service is needed. The TLS fixture requires the OpenSSL
CLI and uses a test CA with certificate and hostname checks enabled. Only DNS and the route to the fixture are
controlled. Tests cover blocked paths and destinations, DNS rebinding, preserved TLS/Host identity, no proxy or
credential forwarding, redirects, trickled headers/bodies, response bounds, pixel limits, metadata removal,
external-converter rejection, and HTTP 400 responses from both APIs.
