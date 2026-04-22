# Generate self-signed certificates for HTTP/2 (run this from project root):
# openssl req -x509 -newkey rsa:4096 -days 365 -nodes -keyout certs/localhost-key.pem -out certs/localhost.pem -subj '/CN=localhost' -addext 'subjectAltName=DNS:localhost'
#
# Then trust this certificate in your OS/browser or use --insecure flag with curl

