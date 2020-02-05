package itzb.riko.services;

import lombok.extern.slf4j.Slf4j;
import net.sourceforge.tess4j.Tesseract;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.List;

@Service
@Slf4j
public class OCRService {

    private Tesseract tesseract;
    private String dataConfigPath;

    @PostConstruct
    private void init() {
        this.tesseract = new Tesseract();
        this.dataConfigPath = getConfigPath();
        this.tesseract.setDatapath(getConfigPath());
    }

    @PreDestroy
    private void cleanUp() {
        try {
            FileUtils.deleteDirectory(new File(this.dataConfigPath));
        } catch (IOException e) {
            log.error("Failed to clean up tesseract config files");
        }
    }

    public String findText(File file) {
        String result = null;
        try {
            result = tesseract.doOCR(file);
            file.deleteOnExit();
        } catch (Exception e) {
            log.error("Error detecting text from file");
        }
        return result;
    }

    private String getConfigPath() {
        String configPath = null;
        try {
            File tessdata = new File("tessdata");
            tessdata.mkdirs();
            File configs = new File(tessdata.getAbsolutePath() + "/configs");
            configs.mkdir();
            List<String> configsFiles = Arrays.asList("api_config", "digits", "hocr");
            List<String> tessdataFiles = Arrays.asList("deu.traineddata", "eng.traineddata", "osd.traineddata", "pdf.ttf", "pdf.ttx");
            for (String configsFile : configsFiles) {
                createTmpFile(configsFile, configs.getAbsolutePath() + "/" + configsFile);
            }
            for (String tessdataFile : tessdataFiles) {
                createTmpFile(tessdataFile, tessdata.getAbsolutePath() + "/" + tessdataFile);
            }
            configPath = tessdata.getAbsolutePath();
        } catch (Exception e) {
            log.error("Can not load tesseract config files");
        }
        return configPath;
    }

    private void createTmpFile(String resourceName, String newPath) throws Exception {
        File file = new File(newPath);
        OutputStream outputStream = new FileOutputStream(file);
        IOUtils.copy(Thread.currentThread().getContextClassLoader().getResourceAsStream(resourceName), outputStream);
    }
}
