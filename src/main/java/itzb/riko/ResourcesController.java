package itzb.riko;

import itzb.riko.services.PersonalDataService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@RequestMapping("/")
public class ResourcesController {
    @Autowired
    private PersonalDataService personalDataService;

    @RequestMapping(method = RequestMethod.GET, path = "/health")
    public String health() {
        return "The server is running and healthy";
    }

    @RequestMapping(method = RequestMethod.POST, path = "/image", produces = MediaType.IMAGE_PNG_VALUE)
    public ResponseEntity<Resource> clearPersonalData(@RequestParam("image") MultipartFile image) throws IOException {
        byte[] bytes = image.getBytes();
        byte[] returnBytes = personalDataService.clearPersonalData(bytes);
        ByteArrayResource resource = new ByteArrayResource(returnBytes);
        return ResponseEntity.ok()
                .body(resource);
    }

    @RequestMapping(method = RequestMethod.POST, path = "/image-person", produces = MediaType.IMAGE_PNG_VALUE)
    public ResponseEntity<Resource> clearFace(@RequestParam("image") MultipartFile image) throws IOException {
        byte[] bytes = image.getBytes();
        byte[] returnBytes = personalDataService.clearFace(bytes);
        ByteArrayResource resource = new ByteArrayResource(returnBytes);
        return ResponseEntity.ok()
                .body(resource);
    }

    @RequestMapping(method = RequestMethod.POST, path = "/image-car", produces = MediaType.IMAGE_PNG_VALUE)
    public ResponseEntity<Resource> clearLicensePlate(@RequestParam("image") MultipartFile image) throws IOException {
        byte[] bytes = image.getBytes();
        byte[] returnBytes = personalDataService.clearLicensePlate(bytes);
        ByteArrayResource resource = new ByteArrayResource(returnBytes);
        return ResponseEntity.ok()
                .body(resource);
    }
}
